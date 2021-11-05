import ee
import tensorflow as tf
import math



def format_image(image, image_bands, response_band, VI_definition):
    image = ee.Image(image)
    image_bands = ee.List(image_bands)
    response_band = ee.String(response_band)
    VI_definition = ee.List(VI_definition)
    
    # image_bands specifices a list of the names of the bands used in defining the expressions for VIs in VI_definition
    image = image.rename(image_bands).toDouble()
    
    # Generate an imageCollection from a list of expressions defining a set of Vegetation Indices using the bands available in the image
    VIimageCollection = ee.ImageCollection(VI_definition.map(lambda expr: image.expression(ee.String(expr))))
    VIimage = VIimageCollection.toBands().regexpRename('[0-9]+_', '')
    
    # Reordering the bands in the image so the response band is the last band in the image
    feature_bands = image_bands.remove(response_band)
    
    return image.select(feature_bands).addBands(VIimage).addBands(image.select(response_band))



def get_num_pixels(image):
    
    # get image height
    def get_height(image):
        height = image.getInfo()["bands"][0]["dimensions"][0]
        return height
    
    # get image width
    def get_width(image):
        width = image.getInfo()["bands"][0]["dimensions"][1]
        return width
    
    image_height = get_height(image)
    image_width = get_width(image)
    image_pixels = image_height*image_width
    
    return image_pixels



def scale_image(image, response_band):
    image = ee.Image(image)
    response_band = ee.String(response_band)
    image_pixels = ee.Number(get_num_pixels(image))
    
    # Setting up lists containing the input/feature bands in the image
    bandList = image.bandNames()
    featureList = bandList.remove(response_band)
    num_bands = bandList.length()
    num_features = featureList.length()
    
    # We will be using the reduceRegion() function on images from Earth Engine, 
    # which will process up to a specified number of pixels from the image to generate the outputs of the reducer
    max_pixels = image_pixels.min(10000000)
    # best_effort = ee.Algorithms.If(image_pixels.gt(max_pixels), True, False)
    
    # Set default projection and scale using the response band
    defaultScale = image.select(response_band).projection().nominalScale()
    defaultCrs = image.select(response_band).projection().crs()
    image = image.setDefaultProjection(crs=defaultCrs, scale=defaultScale)
    
    # Initial centering all of the input bands, added VIs, and response with the input image
    meanImage = image.subtract(image.reduceRegion(reducer=ee.Reducer.mean(), scale=defaultScale, \
                                                  bestEffort=True, maxPixels=max_pixels).toImage(bandList))
    
    # Separating the image into features (X) and response (y) for processing with LARs
    X = meanImage.select(featureList)
    y = meanImage.select(response_band)
    
    # Standardizing the input features
    X = X.divide(X.reduceRegion(reducer=ee.Reducer.stdDev(), bestEffort=True, maxPixels=max_pixels).toImage(featureList))
    
    return X.addBands(y)




def ee_LARS(input_image, input_bandNames, response_bandName, num_nonzero_coefficients, num_samples):
    image = ee.Image(input_image)
    feature_list = ee.List(input_bandNames)
    response_band = ee.String(response_bandName)
    full_band_list = ee.List(feature_list).add(response_band)
    num_nonzero_coefficients = ee.Number(num_nonzero_coefficients)
    num_samples = ee.Number(num_samples)
    image_pixels = ee.Number(get_num_pixels(image))
    
    # Randomly sample pixels in the image at native resolution into a FeatureCollection
    input_collection = image.sample(numPixels=num_samples.min(image_pixels))
    n = input_collection.size()
    m = feature_list.length()
    
    # Use an aggregate array function over the FeatureCollection and map the function over each feature in the band list
    # to generate a dictionary of all of the samples retrieved
    inputs = ee.Dictionary.fromLists(full_band_list, full_band_list.map(lambda feature: input_collection.aggregate_array(feature)))
    
    # Although we may call our scale_image function on the input image, the reduceRegion() function used to determine the mean
    # and standard deviation of each band in the image over the entire region is not precise enough over a large image
    # so we must recenter all of the bands in the image and now we can also normalize (L2 norm) each input feature as required
    # by the LARs algorithm
    
    # Use an aggregate_mean function over the feature collection to get the mean of each band
    input_means = ee.Dictionary.fromLists(full_band_list, full_band_list.map(lambda feature: input_collection.aggregate_mean(feature)))

    def centre_inputs(key, value):
        key_mean = input_means.getNumber(key)
        return ee.List(value).map(lambda sample: ee.Number(sample).subtract(key_mean))
    
    
    # Center bands by mapping over the list of features and then a subtracting over the list of samples for each band
    inputs = inputs.map(centre_inputs)

    # Separate the response variable samples into its own vector
    y = inputs.toArray([response_band]).reshape([-1,1])

    # Remove response band from the feature collection by selecting only bands in the feature list
    inputs = inputs.select(feature_list)
    
    # Generate a dictionary of all of the L2 norms of the input features using a custom mapped function
    input_norms = inputs.map(lambda key, value: ee.Number(ee.List(value).map(lambda sample: ee.Number(sample).pow(2)).reduce(ee.Reducer.sum())).pow(0.5))

    def norm_inputs(key, value):
        key_norm = input_norms.getNumber(key)
        return ee.List(value).map(lambda sample: ee.Number(sample).divide(key_norm))
    
    # Normalize all of the features by mapping a function over the list of features
    # and then map a division over the list of all of the samples of the feature
    inputs = inputs.map(norm_inputs)
    
    # Generate the array of samples using the dictionary
    X = inputs.toArray(feature_list).transpose()

    # Find the first best predictor of the response to initialize the main LARs loop
    initial_prediction = ee.Array(ee.List.repeat([0], n))
    c = X.transpose().matrixMultiply(y.subtract(initial_prediction))
    c_abs = c.abs()
    C_maxLoc = c_abs.project([0]).argmax()
    add_feature = C_maxLoc.getNumber(0)
    A = ee.List([add_feature])
    
    # Create a dicitionary of initial inputs to pass into the main LARs iterative loop
    # The iterate function in Earth Engine processes each iteration as a tree of iterations with no access to any variables
    # from previous iterations (only those that are passed to the next iteration)
    # so we must pass both the current prediction and the active set of features (with non-zero coefficients), A
    initial_inputs = ee.Dictionary({'prediction': initial_prediction, 'A': A})

    def LARs_regression(iteration, inputs):
        inputs = ee.Dictionary(inputs)

        # Find the active set of features, A (predictors with non-zero coefficients)
        A = ee.List(inputs.get('A'))
        # A_list is an array used to mask the full array of input samples and the correlation vector
        A_list = ee.Array(ee.List.sequence(0, m.subtract(1))\
                          .map(lambda index: A.contains(index)).replaceAll(False, 0).replaceAll(True, 1)).reshape([-1,1])

        # The following matrix algebra determines the next most correlated variable, or the next best predictor considering the
        # current features in the active set, A, as well as the magnitude to adjust the prediction vector to ensure all of the
        # features in the active set are equally correlated to response vector
        prediction = inputs.getArray('prediction')
        c = X.transpose().matrixMultiply(y.subtract(prediction))
        c_abs = c.abs()
        C_max = c_abs.get(c_abs.argmax())
        s_A = c.divide(c_abs).mask(A_list)
        X_A = X.mask(A_list.transpose())
        G_Ai = X_A.transpose().matrixMultiply(X_A).matrixInverse()
        G1 = G_Ai.matrixMultiply(s_A)
        A_A = s_A.project([0]).dotProduct(G1.project([0])).pow(-0.5)
        w_A = G1.multiply(A_A)
        u_A = X_A.matrixMultiply(w_A)
        a = X.transpose().matrixMultiply(u_A)
        a = a.project([0])
        c = c.project([0])

        def compute_gammaArray(index_j):
            minus_j = C_max.subtract(c.get([index_j])).divide(A_A.subtract(a.get([index_j])))
            plus_j = C_max.add(c.get([index_j])).divide(A_A.add(a.get([index_j])))
            return ee.List([minus_j, plus_j]).filter(ee.Filter.gte('item', 0)).reduce(ee.Reducer.min())

        A_c = ee.List.sequence(0, m.subtract(1)).removeAll(A)
        gammaArray = A_c.map(compute_gammaArray)
        gamma = gammaArray.reduce(ee.Reducer.min())
        min_location = gammaArray.indexOf(gamma)
        add_feature = A_c.getNumber(min_location)

        # Update active set of variables with next best predictor from non-active set and update prediction vector
        A = A.add(add_feature)
        prediction = prediction.add(u_A.multiply(gamma))

        return ee.Dictionary({'prediction': prediction, 'A': A})


    # The final iteration of LARs (if selecting all input variables) requires a different method to determine magnitude for
    # adjusting the magnitude of the prediction vector, as the regular LARs iteration relies on variables in non-active set
    # In the final iteration there will be no variables in the non-active set, so the method will not work
    def LARs_final_iteration(iteration, inputs):
        inputs = ee.Dictionary(inputs)
        A = ee.List(inputs.get('A'))

        prediction = inputs.getArray('prediction')
        c = X.transpose().matrixMultiply(y.subtract(prediction))
        c_abs = c.abs()
        C_max = c_abs.get(c_abs.argmax())        

        s_A = c.divide(c_abs)
        G_Ai = X.transpose().matrixMultiply(X).matrixInverse()
        G1 = G_Ai.matrixMultiply(s_A)
        A_A = s_A.project([0]).dotProduct(G1.project([0])).pow(-0.5)
        w_A = G1.multiply(A_A)
        u_A = X.matrixMultiply(w_A)

        gamma = C_max.divide(A_A)
        prediction = prediction.add(u_A.multiply(gamma))

        return ee.Dictionary({'prediction': prediction, 'A': A})

    # Actually carrying out the iterations by iterating over a placeholder list (sequence from 1 to the number of non-zero
    # variables that the user wishes to select as predictors for the response)
    iterations = ee.List.sequence(1, m.subtract(1).min(num_nonzero_coefficients))
    penultimate_outputs = iterations.iterate(LARs_regression, initial_inputs)
    final_outputs = ee.Dictionary(ee.Algorithms\
                    .If(num_nonzero_coefficients.gte(m), LARs_final_iteration(m, penultimate_outputs), penultimate_outputs))
    
    final_prediction = final_outputs.getArray('prediction')

    A = ee.List(final_outputs.get('A'))

    feature_path = A.slice(0, num_nonzero_coefficients).map(lambda index: feature_list.getString(index))

    # The code snippet below is able to extract the exact coefficients on all of the selected features, but is commented out
    # as it adds computational complexity that takes up unnecessary memory on the Google Earth Engine virtual machine since we
    # are only using LARs as a feature selection algorithm

#     coefficients = X.matrixSolve(final_prediction).project([0])\
#                               .toList().map(lambda num: ee.Algorithms.If(ee.Number(num).abs().lt(0.001), 0, num))
#     print('Coefficients')
#     coeff = ee.Dictionary.fromLists(featureList, coefficients).getInfo()
#     ordered_coeff = OrderedDict()
#     var_path = feature_path.cat(featureList.removeAll(feature_path)).getInfo()
#     for key in var_path:
#         ordered_coeff[key] = coeff[key]
#     print(json.dumps(ordered_coeff, indent=1))
    
    print('Selected features: ', feature_path.getInfo())
    
    return feature_path



# The following function trims input data according to an algorithm in which the response band is partitioned into n equally sized
# partitions, and in each of the n partitions, for the features selected by LARs, they are each trimmed individually down to only the
# 5-95 percentile of the data. We are not doing any preprocessing with the data, so the raw data is exported from Earth Engine
# The function takes an image, a list of strings with the selected feature bands in the image, the string that is the name of the response
# band in this image, the number of samples/pixels the user wants to take from the image, and the number of parititions to trim within
def trim_data(image, selected_features, response_band, num_samples, num_partitions):
    image = ee.Image(image)
    selected_features = ee.List(selected_features)
    response_band = ee.String(response_band)
    num_samples = ee.Number(num_samples)
    num_partitions = ee.Number(num_partitions)
    
    # Generate the list of percentile bounds for the requested number of partitions, and the names of the value bounds for the
    # dictionary that will be generated from the percentile reducer used later on
    percentiles = ee.List.sequence(0, 100, ee.Number(100).divide(num_partitions))
    percentile_names = percentiles.map(lambda num: ee.Number(num).round().toInt().format("p%s"))

    # Randomly sample the pixels in the input image into a feature collection containing only the selected features and the response
    image_pixels = ee.Number(get_num_pixels(image))
    inputsCollection = image.select(selected_features.add(response_band)).sample(numPixels=num_samples.min(image_pixels))

    # Find the values at the percentile bounds using the percentile reducer over the feature collection
    response_percentiles = inputsCollection.reduceColumns \
                            (ee.Reducer.percentile(percentiles=percentiles, outputNames=percentile_names,
                                                   maxRaw=inputsCollection.size()),[response_band])
    
    # Create a list of percentile bounds for each partition
    response_partitions = response_percentiles.values(percentile_names.remove('p100')).zip\
                                    (response_percentiles.values(percentile_names.remove('p0')))
    
    def partition_data(partition_range):
        partition_range = ee.List(partition_range)
        return inputsCollection.filter(ee.Filter.rangeContains(response_band,
                                                        partition_range.getNumber(0), partition_range.getNumber(1)))
    
    partitioned_data = response_partitions.map(partition_data)

    # The following function now trims the data in each partition individually for each feature to its 5-95 percentile only
    def trim_partitions(partition):
        partition = ee.FeatureCollection(partition)
        feature_trimming_bounds = selected_features\
            .map(lambda feature: ee.List([feature])\
                 .cat(partition.reduceColumns(ee.Reducer.percentile([5, 95]), [feature]).values(['p5','p95'])))
        
        def trimmer(current_feature, collection):
            current_feature = ee.List(current_feature)
            collection = ee.FeatureCollection(collection)
            return collection.filter(ee.Filter.rangeContains(current_feature.getString(0),
                                                             current_feature.getNumber(1),
                                                             current_feature.getNumber(2)))
        
        return feature_trimming_bounds.iterate(trimmer, partition)
    
    # Retrieve the trimmed data partitions and flatten the paritions into a single trimmed feature collection
    trimmed_partitions = partitioned_data.map(trim_partitions)
    trimmed_data = ee.FeatureCollection(trimmed_partitions).flatten()
    
    return trimmed_data



def parse_layer(feature):
    feature = ee.Feature(feature)
    prev_layer_size = feature.getNumber("prev_layer_size")
    num_nodes = feature.getNumber("num_nodes")
    node_size = prev_layer_size.subtract(1)
    activation = feature.getString("activation")
    
    node_collection = ee.ImageCollection(ee.List.sequence(1, num_nodes)\
                        .map(lambda node: ee.ImageCollection(ee.List.sequence(ee.Number(node).toInt(), ee.Number(node)\
                                    .toInt().add(node_size.multiply(num_nodes)), num_nodes)\
                        .map(lambda index: ee.Image(feature.getNumber(ee.Number(index).toInt())))).toBands()\
                        .set({"bias": feature.getNumber(ee.Number(node).toInt().add(prev_layer_size.multiply(num_nodes)))})))
    
    return ee.List([node_collection, activation])



def apply_nnet(layer, net_input):
    layer = ee.List(layer)
    net_input = ee.Image(net_input)
    
    layer_nodes = ee.ImageCollection(layer.get(0))
    activation = layer.getString(1)
    
    node_outputs = layer_nodes.map(lambda node: ee.Algorithms.If(activation.compareTo("linear"),
                        softsign(net_input.multiply(node).reduce(ee.Reducer.sum()).add(node.getNumber("bias"))),
        net_input.multiply(ee.Image(node)).reduce(ee.Reducer.sum()).add(ee.Image(node).getNumber("bias")))).toBands()

    return node_outputs



# ---------------
# MATH FUNCTIONS:
# ---------------

def linear(x):
    return ee.Image(x)

def elu(x):
    x = ee.Image(x)
    return ee.ImageCollection([x.mask(x.gte(0)), x.mask(x.lt(0)).exp().subtract(1)]).mosaic()

def softplus(x):
    x = ee.Image(x)
    return x.exp().add(1).log()

def softsign(x):
    x = ee.Image(x)
    return x.divide(x.abs().add(1))

def relu(x):
    x = ee.Image(x)
    return x.max(0.0)

def tanh(x):
    x = ee.Image(x)
    return x.multiply(2).exp().subtract(1).divide(x.multiply(2).exp().add(1))

def sigmoid(x):
    return x.exp().pow(-1).add(1).pow(-1)



