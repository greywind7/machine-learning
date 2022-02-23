import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

# Support Vector regression needs scaling because its geometric
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
# We reshape because scaling needs a 2d matrix
y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc_y = StandardScaler()
# we need two objects because when fit, its associated
x = sc.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(x,y)

pred_val = sc_y.inverse_transform(regressor.predict(sc.transform([[6.5]])).reshape(1,-1))
print(pred_val)

print(sc.inverse_transform(x))
print(regressor.predict(x).reshape(1,10))
plt.scatter(sc.inverse_transform(x),sc_y.inverse_transform(y) ,color='green')
plt.plot(sc.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(10,1)),color='blue')
plt.title('Salary vs level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()