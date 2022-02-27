from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
(x_train,x_test, y_train,y_test) = train_test_split(x,y,test_size = 0.25, random_state = 0)

# Feature scaling is must in geometry based algorithms
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifier = GaussianNB()
classifier.fit(x_train,y_train)

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(x_train,y_train)

rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(x_train,y_train)

y_pred = classifier.predict(sc.transform(x_test))
y_pred_dt = dt_classifier.predict(sc.transform(x_test))
y_pred_rf = rf_classifier.predict(sc.transform(x_test))

from sklearn.metrics import confusion_matrix,accuracy_score
print("Naive Bayes")
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("Decision Tree")
print(confusion_matrix(y_test,y_pred_dt))
print(accuracy_score(y_test, y_pred_dt))

print("Random Forest")
print(confusion_matrix(y_test,y_pred_rf))
print(accuracy_score(y_test, y_pred_rf))