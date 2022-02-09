from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
(x_train,x_test, y_train,y_test) = train_test_split(x,y,test_size = 0.2, random_state = 69)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)

# Creating a plot of the linear regression line and scaterplot of test set
plt.scatter(x_test,y_test, color='green')
plt.plot(x_train,y_pred,color='red')
plt.title('Salary vs experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
