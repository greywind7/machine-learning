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

from sklearn.svm import LinearSVC, SVC

# For a general SVC, use SVC class with a different kernel
classifier = LinearSVC(random_state=0)
classifier.fit(x_train,y_train)

ker_classifier = SVC(kernel='rbf', random_state=0)
ker_classifier.fit(x_train,y_train)

y_pred = classifier.predict(sc.transform(x_test))
y_pred_ker = ker_classifier.predict(sc.transform(x_test))
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score
print("Linear SVM")
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("RBF kernel SVM")
print(confusion_matrix(y_test,y_pred_ker))
print(accuracy_score(y_test, y_pred_ker))