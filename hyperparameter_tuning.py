# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:47:32 2017

@author: James
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

#PIMA Indians
df = pd.read_csv('pima-indians-diabetes.csv')
column_names = ['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
                'dpf', 'age', 'diabetes']

df.columns = column_names

y = df.diabetes#.values

X = df.drop('diabetes', axis=1)#.values

param_grid = {'n_neighbors': np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, param_grid, cv=5)

knn_cv.fit(X, y)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

#on voting data
#preprocess voting data to generate target and values
voting = pd.read_csv('house-votes-84.csv', header=None, na_values='?').dropna()
voting.replace({'n':0,'y':1}, inplace=True)

y2 = voting[0]#.values#target
X2 = voting.drop(voting.columns[0], axis=1)#.values

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X2,y2)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))

#Hyperparameter tuning with RandomizedSearchCV
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
