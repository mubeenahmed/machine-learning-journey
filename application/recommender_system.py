# Example from book Data Science from scratch first principle with Python By Joel Grus
# Chapter 22

import collections
import numpy as np
import math

user_interests = [
	["Hadoop", "Java", "Spark", "HBase", "Scala", "Big Data"],
	["NoSQL", "MongoDB", "Cassendra", "MySQL", "PostgreSQL", "HBase"],
	["Python", "Java", "Ruby", "Clojure", "Scala", "Erlang"],
	["scipy", "numpy", "R", "Python", "machine learning", "regression"],
	["statistics", "machine learning", "regression", "Hadoop", "algorithms", "Big Data"],
	["depp learning", "artificial intelligence", "probability", "regression", "time series", "modeling"],
	["C++", "Java", "artificial intelligence", "scipy", "sckit-learn", "support vector machines"],
	["neural networks", "feed forward", "back propagation", "convolution networks", "tensorflow", "R"],
	["pandas", "R", "Python"],
	["Hadoop", "Java", "Spark"],
	["HBase", "Scala", "Big Data"],
	["mathematics","programming"]
]

# Recommending the most popular
# Finding most popular interest
# Flatting the array
popular_interest = collections.Counter(interest 
							for user_interests in user_interests
							for interest in user_interests).most_common()

# The user with no/least popular interest, this can recommend 
# For example ["NoSQL", "MongoDB", "Cassendra", "MySQL", "PostgreSQL", "HBase"] 

def most_popular_new_interests(user_interests, max_results = 5):
	suggestions = [ (interest, frequency) 
					for interest,frequency in popular_interest
					if interest not in user_interests]
	return suggestions[:max_results]

my_interest = ["C++", "Java", "artificial intelligence", "scipy", "sckit-learn", "support vector machines"]
most_popular = most_popular_new_interests(my_interest)
print(most_popular)


# User based collaboration filtering (Cosine Similarity)
def cosine_similarity(v, w):
	return np.dot(v, w) / math.sqrt(np.dot(v, v) * np.dot(w, w))

# Get the unique interest list
unique_interest = sorted(list({ interest for user_interests in user_interests
										 for interest in user_interests}))

# Create a vector for the user, iterate over unique interest and subsititue 
# with 1 if found, otherwise 0

def make_user_interest_vector(user_interests):
	return [1 if interest in user_interests else 0 
				for interest in unique_interest]

# Maping to 1 and 0 for all the user interests
user_interests_matrix = map(make_user_interest_vector, user_interests)

user_similarities = [ [cosine_similarity(interest_i, interest_j) 
						for interest_i in user_interests_matrix
						for interest_j in user_interests_matrix] ]

# user_similarities[i] is the vector for every other users
def most_similar_users_to(user_id):
	pairs = [ (other_user_id, similarity) 
			for other_user_id, similarity in enumerate(user_similarities[user_id])
			if user_id != other_user_id and similarity > 0]
	return pairs

p = most_similar_users_to(0)
print(p)

# Suggest user 
def user_based_suggestions(user_id):
	suggestions = collections.defaultdict(float)
	for other_user_id, similarity in most_similar_users_to(user_id):
		for interest in user_interests[other_user_id]:
			suggestions[interest] += similarity

	suggestions = sorted(suggestions.items(), reverse=True)
	return [(suggestion, weight) 
				for suggestion, weight in suggestions 
				if suggestion not in user_interests[user_id]]

final = user_based_suggestions(0)
print(final)