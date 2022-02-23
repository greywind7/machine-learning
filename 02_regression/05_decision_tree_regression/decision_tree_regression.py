import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x,y)
print(regressor.predict([[6.5],[7.5]]))

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color='green')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Salary vs level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()