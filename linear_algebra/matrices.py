
# A matrix is a two dimensional collectons of numbers.

# A has 2 rows and 3 cols
A = [[1,2,3]
	[4,5,6]]


def dimension(A):
	num_rows = len(A)
	num_cols = len(A[0])
	return num_rows, num_cols

def get_rows(A, i):
	return A[i]


def get_cols(A, j):
	return [A_i[j] for A_i in A]

