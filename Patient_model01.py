# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:14:17 2018

@author: amit.jain
"""

# Importing the libraries
import numpy as np
import pandas as pd
import os


os.chdir('E:/Learning Center/PGPBA/Capstone Project/Patient Data')

dspat = pd.read_csv('dspat.csv')
dspat = dspat.drop('Patient_Number', axis=1, inplace=False)

x = dspat.drop('Blood_Pressure_Abnormality', axis=1, inplace=False)
y = pd.DataFrame(dspat.loc[:, ['Blood_Pressure_Abnormality']].values)


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 0)

##########################
### Feature Scaling - This is required because some data elements can contain large values and some data elements can contain small
###    values. When calculating Eucledian distances, large values can dominate the small values thereby making the data elements 
###    containing the smaller values seem insignificant to the model
##########################
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)

##########################
## Fitting the Random Forest Regression model to the training dataset
##########################
from sklearn.ensemble import RandomForestRegressor
randomf_model = RandomForestRegressor(n_estimators = 11, random_state = 0)
randomf_model.fit(x_train, y_train)
print(randomf_model)



