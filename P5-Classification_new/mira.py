# mira.py
# -------
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


# Mira implementation
import util
PRINT = True

def bestLabel(feature, weights, legalLabels):
    """
    Find the label yprime such that weights[yprime] * feature gives highest score 
    on all label.
    feature: An util.Counter
    weights: a list of util.Counter
    legalLabels: a list of legal labels
    """
    assert(len(legalLabels) == len(weights))
    bestLabel = None
    scoreOfBestLabel = None
    for label in legalLabels:
        score = feature * weights[label]
        if bestLabel is None or score > scoreOfBestLabel:
            bestLabel = label
            scoreOfBestLabel = score
    return bestLabel

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        bestWeightsSoFar = None
        accuracyForBestWeightsSoFar = None

        # train with each C and see how well the classifier did on the validation set
        for C in Cgrid:
            weights = self.weights.copy()
            # training stage
            for iteration in range(self.max_iterations):
                for i in range(len(trainingData)):
                    feature = trainingData[i]
                    y = trainingLabels[i]
                    # find the predicted label for this sample
                    yprime = bestLabel(feature, weights, self.legalLabels)

                    # correct prediction. Good!
                    if yprime == y: continue
                    # wrong prediction. Correct it!
                    tau = ((weights[yprime] - weights[y]) * feature + 1.0) / (2.0 * (feature * feature))
                    tau = min(C, tau)

                    delta = feature.copy()
                    delta.divideAll(1.0 / tau)
                    weights[y] += delta
                    weights[yprime] -= delta

            # testing (on validation) stage
            accuracy = 0.
            for i in range(len(validationData)):
                feature = validationData[i]
                y = validationLabels[i]
                yprime = bestLabel(feature, weights, self.legalLabels)

                if yprime == y: accuracy += 1

            if bestWeightsSoFar is None or accuracy > accuracyForBestWeightsSoFar:
                bestWeightsSoFar = weights
                accuracyForBestWeightsSoFar = accuracy
            print("Performance on validation set for C=" + str(C) + ": (" + str(accuracy/len(validationData)) + ")")

        self.weights = bestWeightsSoFar


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        sortedKeys = self.weights[label].sortedKeys()
        assert(len(sortedKeys) >= 100)
        featuresWeights = sortedKeys[0:100]

        return featuresWeights