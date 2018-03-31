
# Using multivarient classification dataset http://archive.ics.uci.edu/ml/datasets/Heart+Disease
# Using http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data

# First step is know your data. Take your time in knowing your data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

head = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
		'restecg', 'thalach', 'exange', 'oldpeak', 'slope', 
		'ca', 'thal', 'y']

dataframe = pd.read_csv('heart-disease.csv', sep=' ', names = head)
dataframe = dataframe.dropna()

print(dataframe.shape)
print(list(dataframe.columns))

# Merging 1,2,3 classes into 1
u = dataframe['y'].unique()
print(u)
dataframe['y'] = np.where(dataframe['y'] == 2, 1, dataframe['y'])
dataframe['y'] = np.where(dataframe['y'] == 3, 1, dataframe['y'])
dataframe['y'] = np.where(dataframe['y'] == 4, 1, dataframe['y'])

u = dataframe['y'].unique()
print(u)

mean_by_y = dataframe.groupby('y').mean()
mean_by_sex = dataframe.groupby('sex').mean()
mean_by_age = dataframe.groupby('age').mean()

pd.crosstab(dataframe.age, dataframe.y).plot(kind = 'bar')
plt.title('Frequency per age')
plt.xlabel('Age')
plt.ylabel('Disease')

pd.crosstab(dataframe.sex, dataframe.y).plot(kind = 'bar')
plt.title('Frequency per sex')
plt.xlabel('Sex')
plt.ylabel('Disease')
# plt.show()

print(mean_by_y)
print(mean_by_sex)
print(mean_by_age)

dataframe['y'].value_counts()
sns.countplot(x = 'y', data = dataframe, palette='hls')
# plt.show()

y = ['y']
X=[i for i in dataframe if i not in y]

# Using RFE
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split

logreg = LogisticRegression()
rfe = RFE(logreg, 14)
rfe = rfe.fit(dataframe[X], dataframe.y )

print(rfe.support_)
print(rfe.ranking_)

# Since all is true, let select all
col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
		'restecg', 'thalach', 'exange', 'oldpeak', 'slope', 
		'ca', 'thal']

X=dataframe[col]
y=dataframe['y']

# import statsmodels.api as sm

# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# Reference:
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8