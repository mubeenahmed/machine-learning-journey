from collections import defaultdict
from collections import Counter


inputs = [
	
	({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False ),
	({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False ),
	({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True ),
	({'level': 'Junior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, True ),
	({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, False ),
	({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True ),
	({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True ),
	({'level': 'Senior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'yes'}, True ),
]

# Entropy function
def entropy(class_probab):
	return sum( -p * math.log(p, 2) for p in class_probab if p)

# Probabilities by class
def class_probab(labels):
	total_counts = len(labels)
	return [count / total_counts for count in Counter(labels).values()]


def data_entropy(labeled_data):
	labels = [ label for _, label in labeled_data]
	probabilities = class_probab(labels)
	return probabilities

def partition_entropy(subsets):
	total_counts = sum(len(subset) for subset in subsets)
	return sum( data_entropy(subset) * len(subset) / total_counts for subset in subsets)

# Get the entropy
def partition_by(inputs, attribute):
	groups = defaultdict(list)
	for input in inputs:
		key = input[0][attribute]
		groups[key].append(key)
	return groups


def partition_by_entropy(inputs, attribute):
	partitions = partition_by(inputs, attribute)
	return partition_entropy(partitions.values())



for key in ['level', 'lang', 'tweets', 'phd']:
	partition_by_entropy(inputs, key)


from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

graph = graphviz.Source(dot_data)
