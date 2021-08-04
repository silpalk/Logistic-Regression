# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:52:36 2021

@author: asilp
"""


import pandas as pd
import numpy as np
bank_data=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign11\bank_data.csv")
bank_data.columns
bank_data.dtypes
bank_data.describe()


x=bank_data.iloc[:,:31]
y=bank_data.iloc[:,[31]]
import matplotlib.pyplot as plt
import seaborn as sns
##graphical representation of data
%matplotlib inline
bank_data['age'].plot.hist(bins=30)
sns.jointplot(x='age',y='balance',data=bank_data)
sns.jointplot(x=ad_data['Daily_Time_ Spent _on_Site'],y=ad_data['Daily Internet Usage'], data=ad_data, color = 'r')
sns.pairplot(data=bank_data.iloc[:200,:],hue='y')

##model building

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

##training the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_test
y_predict=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_predict)
model.score(x_train,y_train)
confusion_matrix(y_test,y_predict)
##Accuracy of model = (TP+TN)/(TP + TN + FP + FN)
accuracy=(7792+290)/(7792+166+795+290)
accuracy
