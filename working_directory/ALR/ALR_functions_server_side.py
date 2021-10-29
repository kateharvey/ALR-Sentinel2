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




def EE_LARS(image, response_band, num_nonzero_coefficients, num_samples):
    image = ee.Image(image)
    response_band = ee.String(response_band)
    num_nonzero_coefficients = ee.Number(num_nonzero_coefficients)
    num_samples = ee.Number(num_samples)
    bandList = image.bandNames()
    featureList = bandList.remove(response_band)
    image_pixels = get_num_pixels(image)
    inputCollection = image.sample(None, None, None, None, num_samples.min(image_pixels), 0, True, 1, True)
    
    n = inputCollection.size()
    m = featureList.length()
    
    inputs = ee.Dictionary.fromLists(bandList, bandList.map(lambda feature: inputCollection.aggregate_array(feature)))
    
    # Re-centering the features and response as reduceRegion() is not as precise as centering with arrays 
    input_means = ee.Dictionary.fromLists(bandList, bandList.map(lambda feature: inputCollection.aggregate_mean(feature)))
    
    def center_inputs(key, value):
        key_mean = input_means.getNumber(key)
        return ee.List(value).map(lambda sample: ee.Number(sample).subtract(key_mean))
    
    inputs = inputs.map(center_inputs)
    y = inputs.toArray([response_band]).reshape([-1,1])

    # Re-normalizing the input features as reduceRegion() is not as precise as normalizing with arrays
    inputs = inputs.select(featureList)
    input_norms = inputs.map(lambda key, value: ee.Number(ee.List(value)\
                        .map(lambda sample: ee.Number(sample).pow(2)).reduce(ee.Reducer.sum())).pow(0.5))
    
    def norm_inputs(key, value):
        key_norm = input_norms.getNumber(key)
        return ee.List(value).map(lambda sample: ee.Number(sample).divide(key_norm))
    
    inputs = inputs.map(norm_inputs)
    X = inputs.toArray(featureList).transpose()
    
    # Find the first most correlated variable to pass into the main loop
    initial_prediction = ee.Array(ee.List.repeat([0], n))
    c = X.transpose().matrixMultiply(y.subtract(initial_prediction))
    c_abs = c.abs()
    C_maxLoc = c_abs.project([0]).argmax()
    add_feature = C_maxLoc.getNumber(0)
    A = ee.List([add_feature])
    
    initial_inputs = ee.Dictionary({'prediction': initial_prediction, 'A': A})
    # print(A.getInfo())
    # print(initial_inputs.getInfo())
    
    # Main loop
    def LARs_regression(iteration, inputs):
        inputs = ee.Dictionary(inputs)
        A = ee.List(inputs.get('A'))
        A_list = ee.Array(ee.List.sequence(0, m.subtract(1)).map(lambda index: A.contains(index))\
                          .replaceAll(False, 0).replaceAll(True ,1)).reshape([-1,1])

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
        A = A.add(add_feature)
        prediction = prediction.add(u_A.multiply(gamma))

        return ee.Dictionary({'prediction': prediction, 'A': A})

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
    
    iterations = ee.List.sequence(1, m.subtract(1).min(num_nonzero_coefficients))
    penultimate_outputs = iterations.iterate(LARs_regression, initial_inputs)
    final_outputs = ee.Dictionary(ee.Algorithms.If(num_nonzero_coefficients.gte(m), 
                                                      LARs_final_iteration(m, penultimate_outputs),
                                                      penultimate_outputs))
    
    final_prediction = final_outputs.getArray('prediction')
    A = ee.List(final_outputs.get('A'))
    feature_path = A.slice(0, num_nonzero_coefficients).map(lambda index: featureList.getString(index))
    
    coefficients = X.matrixSolve(final_prediction)
    
    def set_zero(num):
        num = ee.Number(num)
        return ee.Algorithms.If(num.abs().lt(0.001), 0, num)
    
    coefficients = coefficients.project([0]).toList().map(lambda num: set_zero(num))
    # print('Coefficients', ee.Dictionary.fromLists(featureList, coefficients))
    
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



