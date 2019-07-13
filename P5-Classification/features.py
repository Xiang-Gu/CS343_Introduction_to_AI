# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractor(datum)

    whitespaces = numConnectedWhiteSpaces(datum)
    whitespacesFeatures = oneHotEncode(8, whitespaces)

    features = np.append(features, whitespacesFeatures)
 

    #this part of code print of the number and its numConnectedWhitleSpaces and the one-hot encoding of the result

    # pattern = np.zeros_like(datum, dtype=int)
    # pattern[datum > 0] = 1
    # pattern = features.flatten()

    # print "-" * (datum.shape[1] + 2)
    # s = '|'
    # for i in range(len(pattern)):
    #     if pattern[i] == 0:
    #         s += ' '
    #     else:
    #         s += '#'
    #     if i != 0 and i % datum.shape[1] == 0:
    #         print s + '|'
    #         s = '|'
    # print "-" * (datum.shape[1] + 2)
    # print "WHITESPACES:" + str(whitespaces)
    # print whitespacesFeatures

    return features

def oneHotEncode(size, value):
    """
    size - number of possible values i.e. 1-8 -> 8
    value - value we want to encode i.e. 4 -> [0, 0, 0, 1, 0, 0, 0, 0]
    """
    result = [0]*(size)
    result[value-1] = 1
    return result

def numConnectedWhiteSpaces(datum):
    """
    Determine the number of continuous whitespaces. In other 
    words, number of loops + the outside area
    """
    rows = datum.shape[0]
    cols = datum.shape[1]
    visited = []
    neighbors = []
    result = 0

    for i in range(rows):
        for j in range(cols):
            #currentPixel is 0 (white) or non-zero (black)
            currentPixel = datum[i][j]
            if (i,j) not in visited:
                visited.append((i,j))
                if currentPixel == 0:
                    neighbors.extend(getNeighbors(i, j, rows, cols))
                    result += 1
                    # Evaluate all neighbors that are white pixels and mark all as visited
                    while len(neighbors) > 0:
                        coords = neighbors.pop(0)
                        if coords not in visited:
                            visited.append((coords[0], coords[1]))
                            if datum[coords[0]][coords[1]] == 0:
                                neighbors.extend(getNeighbors(coords[0], coords[1], rows, cols))

    return result



def getNeighbors(i, j, rows, cols):
    """
    return a list of tuples (of indices) of the neighbors of (i, j) in a board with rows rows and cols columns
    """
    result = []
    if i > 0:
        result.append((i - 1, j))
    if j > 0:
        result.append((i, j - 1))
    if i < rows - 1:
        result.append((i + 1, j))
    if j < cols - 1:        
        result.append((i, j + 1))

    return result

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
