# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:53:43 2017

@author: James
"""
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

#White wine quality data centering and scaling
df = pd.read_csv('white-wine.csv')

#bin quality to binary
bins = [0,5,10] #includes 5 in the first bin
group_names = [0,1]
df.quality = pd.cut(df.quality, bins, labels=group_names).astype(int)

X = df.drop('quality', axis=1)
y= df.quality

# Setup the pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))