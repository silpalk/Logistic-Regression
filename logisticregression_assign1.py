# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 12:30:20 2021

@author: Amarnadh Tadi
"""

import pandas as pd
import numpy as np
data=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign11\Affairs.csv")        
data.columns
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data['naffairs'] = np.where(data['naffairs']>0,1,np.where(data['naffairs']<=0,0,1))
import matplotlib.pyplot as plt
import seaborn as sns
##graphical representation of data
%matplotlib inline
data['kids'].plot.hist(bins=30)

sns.jointplot(x='hapavg',y='yrsmarr2',data=data)

sns.pairplot(data=data.iloc[:50,:],hue='naffairs')


x=data.drop(['naffairs'],axis='columns')
y=data.iloc[:,[0]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
x_test.head(5)

y_predict=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
model.score(x_train,y_train)
accuracy_score(y_test,y_predict)
confusion_matrix(y_test,y_predict)
##Accuracy of model = (TP+TN)/(TP + TN + FP + FN)
accuracy=(89+90)/(89+8+13+90)
accuracy

