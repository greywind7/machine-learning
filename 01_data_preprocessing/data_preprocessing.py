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

print(f_mat)
print(dep)

# Taking care of missing data

from sklearn.impute import SimpleImputer
# Using mean as the strategy
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(f_mat[:,1:3])
# Transform returns the updated values
f_mat[:,1:3] = imputer.transform(f_mat[:,1:3])

print(f_mat)
print(dep)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# transformers has 3 arguments, 1st encoder, 2nd encoding function, third, the indexes that need to be encoded
# remainder is for keeping (passthrough) or deleting (drop) the rest of the columns. Defaults to drop
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
f_mat = np.array(ct.fit_transform(f_mat))

print(f_mat)

from sklearn.preprocessing import LabelEncoder

# LabelEncoder only transforms only vectors so no prob
le = LabelEncoder()
dep = le.fit_transform(dep)

print(dep)