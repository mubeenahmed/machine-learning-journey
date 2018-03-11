
import numpy as np
import random
import matplotlib.pyplot as plt

def gradient_descent(x, y, theta, alpha, m, iteration):
	# transposing matrices x to multiple by theta
	# multipling x by theta gives the hypothesis
	# visualization of this step is that if theta = [0, 1] and x = [ [1, 101],[1, 122],[1, 143],[1, 214] ]
	# then the equation is h(theta) = theta0 + X.theta1
	x_trans = x.transpose()
	for i in range(0, iteration):
		hypothesis = np.dot(x, theta)
		# Calculating difference
		loss = hypothesis - y
		# cost function 
		cost = np.sum(loss ** 2) / (2 * m)

		gradient = np.dot(x_trans, loss) / m

		theta = theta - alpha * gradient

	return theta


def generate_data(num_points, bias, variance):
	x = np.zeros(shape=(num_points, 2))
	y = np.zeros(shape=(num_points))

	for i in range(0, num_points):
		x[i][0] = 1
		x[i][1] = i

		y[i] = (i + bias) + random.uniform(0, 1) * variance
	return x, y

def hypothesis(x, theta):
	return np.dot(x,theta)

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = generate_data(100, 25, 10)
m, n = np.shape(x)
numIterations= 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradient_descent(x, y, theta, alpha, m, numIterations)
print(theta)

plt.plot(x, y, 'ro')
plt.plot(x, hypothesis(x, theta), 'orange')
plt.show()