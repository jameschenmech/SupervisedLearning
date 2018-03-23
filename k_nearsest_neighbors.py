# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:50:29 2017

@author: James
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# #Iris Data for KNN
# =============================================================================
iris = datasets.load_iris()
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])

#preprocess voting data to generate target and values
voting = pd.read_csv('house-votes-84.csv', header=None, na_values='?').dropna()
voting.replace({'n':0,'y':1}, inplace=True)

y = voting[0].values#target
X = voting.drop(voting.columns[0], axis=1).values

#create a k-NN classifer with 6 neighbors
#knn = KNeighborsClassifier(n_neighbors=6)

#Fit the classifer to the data
knn.fit(X, y)

#generated random data to test 
X_new = [0.640502,  0.901561,  0.012352,  0.748038,  0.581447,
                      0.633668,  0.930011,  0.088546,  0.464185,  0.445092,
                      0.206788,  0.691602,  0.767457,  0.860979, 0.836291,
                      0.588182]
X_new = pd.DataFrame([X_new]) #convert list to row pandaframe, w/0 brackets defaults to column
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

#Measuring accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                     random_state=21, stratify=y) 
#stratify to keep labels same across test and train
knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))

print(knn.score(X_test, y_test))


# =============================================================================
# #Dataset Digits
# =============================================================================
#Digits recognition dataset
# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.figure()
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
#plt.close()

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))



# =============================================================================
# #Constructing a complexity curve
# =============================================================================
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors): #enumerate adds a counter to iterable
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.figure()
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
