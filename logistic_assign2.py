# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:43:22 2021

@author: asilp
"""

import pandas as pd
import numpy as np
ad_data=pd.read_csv(r"C:\Users\asilp\Desktop\datascience\assign11\advertising.csv")
ad_data.columns
ad_data.dtypes
type(ad_data['Timestamp'][1])
ad_data.drop(['Ad_Topic_Line','Timestamp'],axis=1,inplace=True)
ad_data.drop(['City' , 'Country'],axis=1,inplace=True)

x=ad_data.iloc[:,0:5]
y=ad_data.iloc[:,[5]]
import matplotlib.pyplot as plt
import seaborn as sns
##graphical representation of data
%matplotlib inline
ad_data['Age'].plot.hist(bins=30)
ad_data[ad_data['Clicked_on_Ad']==1]['Country'].value_counts().head(10)
sns.jointplot(x='Age',y='Area_Income',data=ad_data)
sns.jointplot(x=ad_data['Daily_Time_ Spent _on_Site'],y=ad_data['Daily Internet Usage'], data=ad_data, color = 'r')
sns.pairplot(data=ad_data,hue='Clicked_on_Ad')
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
accuracy=(89+90)/(89+8+13+90)
accuracy
