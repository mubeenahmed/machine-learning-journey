import csv
import random
import math

# Reference: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# Using
# https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data

# Step 1: Handle Data
def loadCsv(filename):
	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)

	for i in range(len(dataset)):
		dataset[i] = [ float(x) if type(x).__name__ != 'str' else x for x in dataset[i] ]
	return dataset

# Ratio represent how much to divide dataset between training and testing
def splitDataset(dataset, splitRatio):
	# Calculating size of training size for example 100 * 0.67 = 67 dataset to seperate
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)

	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))

	return [trainSet, copy]

# Seperating the class by M, F, I in my case for classification
def seperateByClass(dataset):
	seperate = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if(vector[0] not in seperate):
			seperate[vector[0]] = []
		seperate[vector[0]].append(vector[1:])
	return seperate

# Calcuate the mean by the class
def mean(numbers):
	n = tuple(map(float, numbers))
	return sum(n)/float(len(n))

# Calculate standard deviation by class
def stddev(numbers):
	n = tuple(map(float, numbers))
	avg = mean(n)
	variance = sum([pow(x - avg,2) for x in n])/float(len(n)-1)
	return math.sqrt(variance)

def summary(dataset):
	summaries = [ (mean(attributes), stddev(attributes)) for attributes in zip(*dataset) ]
	return summaries

# We need to calculate data's means, standard deviation for predictions
def summariesByClass(dataset):
	seperate = seperateByClass(dataset)
	summarize = {}
	for classValue, instances in seperate.items():
		summarize[classValue] = summary(instances)
	return summarize

# Calculating probabilities using PDF
def calculatingProbabilities(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# Some what similar to probabilities
def calculateClassProbabilities(summarize, inputVector):
	probabilities = {}
	for classValue, instances in summarize.items():
		probabilities[classValue] = 1
		for i in range(len(instances)):
			mean, stdev = instances[i]
			x = float(inputVector[i])
			probabilities[classValue] *= calculatingProbabilities(x, mean, stdev)

	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPrediction(summaries, testset):
	predictions = []
	for i in range(len(testset)):
		result = predict(summaries, testset[i][1:])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][0] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def app():
	filename = 'abalone.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	training, test = splitDataset(dataset, splitRatio)
	sumaries = summariesByClass(training)
	predictions = getPrediction(sumaries, test)
	accuracy = getAccuracy(test, predictions)
	print('Accuracy {0} %'.format(accuracy))

app()