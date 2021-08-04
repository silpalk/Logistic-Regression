# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:30:52 2021

@author: asilp
"""


import pandas as pd
import numpy as np
election_data=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign11\election_data.csv")
election_data.columns
election_data.dtypes

election_data.isnull().sum()
election_data=election_data.iloc[1:12,:]
election_data.isnull().sum()


y=election_data.iloc[:,[1]]
x=election_data.drop(['Result'],axis='columns')
##graphical representation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
election_data['Amount Spent'].plot.hist(bins=15)
election_data.dtypes
election_data['Amount Spent'].plot.hist(bins=15)

sns.jointplot(x='Amount Spent',y='Popularity Rank',data=election_data)

sns.pairplot(data=election_data,hue='Result')
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
confusion_matrix(y_test,y_predict)
##Accuracy of model = (TP+TN)/(TP + TN + FP + FN)
accuracy=(0+1)/(0+0+1+1)
accuracy
