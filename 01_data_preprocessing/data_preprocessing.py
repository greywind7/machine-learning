# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets

dataset = pd.read_csv('Data.csv')

# Features are called independent variables
# Dependent variable is what we are trying to predict

# iloc treats the csv as a matrix
f_mat = dataset.iloc[:,:-1].values # Taking all rows and columns except the last one
dep = dataset.iloc[:,-1].values

# Taking care of missing data

from sklearn.impute import SimpleImputer
# Using mean as the strategy
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(f_mat[:,1:3])
# Transform returns the updated values
f_mat[:,1:3] = imputer.transform(f_mat[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# transformers has 3 arguments, 1st encoder, 2nd encoding function, third, the indexes that need to be encoded
# remainder is for keeping (passthrough) or deleting (drop) the rest of the columns. Defaults to drop
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
f_mat = np.array(ct.fit_transform(f_mat))

from sklearn.preprocessing import LabelEncoder

# LabelEncoder only transforms only vectors so no prob
le = LabelEncoder()
dep = le.fit_transform(dep)

# FEATURE SCALING MUST BE DONE AFTER SPLITTING
# This is to ensure that data from the data from test set is not leaked to the training set

from sklearn.model_selection import train_test_split
# first, second arguments are data matrix and dependent variable vector
# test_size gives the percentage of values that go to the test set
# random_state gives the seed
(X_train,X_test, Y_train,Y_test) = train_test_split(f_mat,dep,test_size = 0.2, random_state = 1)

# Standardization 
# (x - mean(x)) / std_dev(x)
# Normalization 
# (x - min(x)) / (max(x) - min(x)) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# do NOT feature scaling on dummy variables
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
# test set must be same scaled as the training set
X_test[:,3:] = sc.transform(X_test[:,3:])

print(X_train,'\n')
print(X_test)
