

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

def column(matrix, i):
    return [row[i] for row in matrix]
    
def split(x, y, ratio):
  return X[:ratio], y[:ratio]

X, y = make_regression(n_samples= 220, n_features= 4, bias=0.0, noise=0.02, shuffle=True)

training_feature_x2 = column(X, 2)[:190]
testing_feature_x2 = column(X, 2)[190:]

training_x2 = np.array(training_feature_x2).reshape(-1, 1)
testing_x2 = np.array(testing_feature_x2).reshape(-1, 1)
training_y = y[:190]
testing_y = y[190:]

lm = LinearRegression()
lm.fit(training_x2, training_y)

predictions_y = lm.predict(testing_x2) 

# The coefficients
print('Coefficients: \n', lm.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(testing_y, predictions_y))
      
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(testing_y, predictions_y))

plt.scatter(training_x2, y, color='g')
plt.plot(testing_x2, predictions_y)

plt.xlabel("X dependent")
plt.ylabel("Y independent")

plt.title("Regression")
plt.show()
