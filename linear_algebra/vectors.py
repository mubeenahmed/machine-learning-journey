
# We don't have to reinvent the wheel, there are libraries 
# such as numpy however, we are writing the logic for learning purposes

# Vectors are object that can be added to form a new vector
# Vectors can be multipled by scalar number to form different
# magnitude vectors

# In machine learning if we had weight, age and height of large 
# number of people, we can consider it has the three 
# dimension of vector

height_weight_age = [70, 170, 40]


# Adding vector
def vector_add(u, w):
	return [v_i + w_i for v_i, w_i in zip(u,w)]

def vector_sub(u, w):
	return [v_i - w_i for v_i, w_i in zip(u,w)]

def vector_sum(vectors):
	result = vectors[0]
	for vector in vectors[1:]:
		result = vector_add(result, vector)
	return result

def vector_sum_v2(vectors):
	return reduce(vector_add, vectors)

def vector_multiply(scalar, v):
	return [scalar * e for e in v]


def vector_mean(vectors):
	n = len(vectors)
	return vector_multiply(1/n, vector_sum(vectors))

a = vector_add([1,2,3], [1,2,3])
b = vector_sub([1,2,3], [1,2,3])

two_vectors = [[1,2,3],[1,2,3],[1,2,3]]
c = vector_sum(two_vectors)
d = vector_multiply(2, [1,2,3])

mean = vector_mean([[1,2,3],[2,3,4]])

print(a)
print(b)
print(c)
print(d)
print(mean)


