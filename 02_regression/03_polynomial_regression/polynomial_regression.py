import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

y_pred = lin_reg.predict(x)

plt.scatter(x,y, color='green')
plt.plot(x,y_pred,color='blue')
plt.title('Salary vs level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

# For polynomial linear regression we use the same feature
# But we take multiple powers of it and then perform multiple linear regression
mul_reg = LinearRegression()
mul_reg.fit(x_poly,y)

y_pred = mul_reg.predict(x_poly)

plt.scatter(x,y, color='green')
plt.plot(x,y_pred,color='blue')
plt.title('Salary vs level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(mul_reg.predict(poly_reg.fit_transform([[6.5]])))
