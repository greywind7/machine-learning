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