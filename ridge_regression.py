# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:50:11 2017

@author: James
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:58:13 2017

@author: James
"""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gm_2008_region.csv')

df_columns = df.drop(['life','Region'], axis=1).columns

# Create arrays for features and target variable
X = df[df_columns.values].values
y = df['life'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=0.3, random_state=42)

# =============================================================================
# #ridge regression penalizes large coefficients
# =============================================================================
ridge = Ridge(alpha=0.1, normalize=True)  #alpha needs to be chosen systematically

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

print('\nRidge regressions score:  ',ridge.score(X_test, y_test))

# =============================================================================
# #lasso regression
# =============================================================================
#lasso regression shrinks the coefficients of less important features to be 
#exactly 0
from sklearn.linear_model import Lasso

boston = pd.read_csv('boston.csv')
y = boston.MEDV.values
X = boston.drop('MEDV', axis=1).values

names = boston.drop('MEDV', axis=1).columns

lasso = Lasso(alpha=0.1) #how do you pick up this coef?



lasso_coef = lasso.fit(X,y).coef_

plt.figure()
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')

plt.show()

# =============================================================================
# #plots R2 as well as standard error for each alpha
# =============================================================================

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)
    

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize = True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv = 10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


