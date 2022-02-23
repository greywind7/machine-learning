import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
(x_train,x_test, y_train,y_test) = train_test_split(x,y,test_size = 0.2, random_state = 0)


# MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
mul_regressor = LinearRegression()
mul_regressor.fit(x_train,y_train)

y_pred = np.array(mul_regressor.predict(x_test))
np.set_printoptions(precision=2)

print("MULTIPLE LINEAR REGRESSION")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("R2 score")

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))
print()

# POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_test)

poly_regressor = LinearRegression()
poly_regressor.fit(x_poly,y_test)

print("POLYNOMIAL REGRESSION")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("R2 score")

y_pred = poly_regressor.predict(x_poly)
print(r2_score(y_test,y_pred))
print()

# SUPPORT VECTOR REGRESSION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc_y = StandardScaler()
y_train_svr = y_train.reshape(len(y_train),1)
x_svr = sc.fit_transform(x_train)
y_svr = sc_y.fit_transform(y_train_svr).reshape(1,len(y_train_svr))
sc_x_test = StandardScaler()
x_svr_test = sc_x_test.fit_transform(x_test)

print("SUPPORT VECTOR REGRESSION")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("R2 score")

from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(x_svr,y_svr[0])
y_pred = sc_y.inverse_transform(svr_regressor.predict(x_svr_test).reshape(1,-1))
print(r2_score(y_test,y_pred[0]))
print()

# DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_train,y_train)
y_pred = dt_regressor.predict(x_test)

print("DECISION TREE REGRESSION")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("R2 score")
print(r2_score(y_test,y_pred))
print()

# RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor 
rf_regressor = RandomForestRegressor(random_state=0, n_estimators=10)
rf_regressor.fit(x_train,y_train)
y_pred = rf_regressor.predict(x_test)

print("RANDOM FOREST REGRESSION")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print("R2 score")
print(r2_score(y_test,y_pred))
print()