

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import make_regression

def column(matrix, i):
    return [row[i] for row in matrix]

X, y = make_regression(n_samples= 220, n_features= 4, bias=0.0, noise=0.3, shuffle=True)

training_feature_x2 = column(X, 2)
testing_feature_x2 = column(X, 2)

training_x2 = np.array(training_feature_x2).reshape(-1, 1)
testing_x2 = np.array(testing_feature_x2).reshape(-1, 1)

lm = LinearRegression()
lm.fit(training_x2, y)

predictions_y = lm.predict(testing_x2) 

plt.scatter(training_x2, y, color='g')
plt.plot(testing_x2, predictions_y)

plt.xlabel("X dependent")
plt.ylabel("Y independent")

plt.title("Regression")
plt.show()