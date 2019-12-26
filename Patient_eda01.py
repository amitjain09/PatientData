# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:16:50 2018

@author: amit.jain
"""
# Importing the libraries
import numpy as np
import pandas as pd
import os

### Following libraries are specifically for graphs, plots and charts
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
#import pylab
import scipy.stats as stats
from pandas.plotting import scatter_matrix


os.chdir('E:/Learning Center/PGPBA/Capstone Project/Patient Data')

dspat = pd.read_csv('Data.csv')

agebins = [0,20,30,40,45,50,60,80]
agelabel = ['0-20', '21-30', '31-40', '41-45', '46-50', '51-60', '61-80']
dspat['Age_group'] = pd.cut(dspat['Age'], bins=agebins, labels=agelabel, include_lowest=True)

## Find missing values
dstt = dspat[(dspat['Level_of_Hemoglobin']==0)]
dstt = dspat[(dspat['BMI']==0)]
dstt = dspat[(dspat['Physical_activity']==0)]
dstt = dspat[(dspat['salt_content_in_the_diet']==0)]
dstt = dspat[(dspat['alcohol_consumption_per_day']==0)]
dstt = dspat[(dspat['Genetic_Pedigree_Coefficient']==0)]

dspat['Sex'].value_counts()
dspat['Pregnancy'].value_counts()

# Finding Null values
len(dspat[(pd.isnull(dspat['Level_of_Hemoglobin']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['BMI']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['Physical_activity']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['salt_content_in_the_diet']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['Sex']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['Genetic_Pedigree_Coefficient']) == True)])  ## This one works

len(dspat[(pd.isnull(dspat['Pregnancy']) == True)])  ## This one works

### Analysis of Null Pregnancy records
dstt = dspat[(pd.isnull(dspat['Pregnancy']) == True)]
dstt['Age_group'].hist()

## Since there are both Males and Females which have pregnancy as NULL. Set Pregnancy = 0 for Males
dspat.loc[dspat.Sex == 0, 'Pregnancy'] = 0
## Set Pregnancy = 0 for Femails where Age >= 45. After this there are no more null Pregnancy records
dspat.loc[dspat.Age >= 45, 'Pregnancy'] = 0

## Finding null values in alcohol_consumption file
dstt = dspat[(pd.isnull(dspat['alcohol_consumption_per_day']) == True)]  ## This one works

### Analysis of Null Alcohol Consumption records
# Based on Pregnancy
dstt['Pregnancy'].hist()

# Redefine Age groups specifically for Alcohol Consumption data imputation
agebins = [0,30,40,45,50,60,80]
agelabel = ['0-30', '31-40', '41-45', '46-50', '51-60', '61-80']
dspat['Age_group'] = pd.cut(dspat['Age'], bins=agebins, labels=agelabel, include_lowest=True)

# Update Alcohol consumption based on Pregnancy and Age_Group
dspat.loc[dspat.Pregnancy==1,'alcohol_consumption_per_day'] = dspat[(dspat['Pregnancy']==1)].groupby(['Age_group'])['alcohol_consumption_per_day'].apply(lambda x: x.fillna(x.mean()))
### Similar code which performs something similar but without Groupby and apply
#df = pd.DataFrame({'c':[10,50,100,200] * 5,
#                   'b':[1,3,8,np.nan,5,8,np.nan,7, 17, 4,1,25   ,6,20,18,15,7,0,4,9],
#                   'a':[1,1,0,0,1,1,1,1,0,0,0,0,  1,0,0,0,1,1,1,1]})
#df.loc[df.a==0,'b'] = df.loc[df.a==0,'b'].fillna(df[df['a'] == 0]['b'].mean())

### Update Null Alcohol Consumption records
## Update the remaining null alcohol_consumption_per_day values based on Sex and Age Group
dspat['alcohol_consumption_per_day'] = dspat.groupby(['Sex', 'Age_group'])['alcohol_consumption_per_day'].apply(lambda x: x.fillna(x.mean()))

# Redefine Age groups as per originally defined.
agebins = [0,20,30,40,45,50,60,80]
agelabel = ['0-20', '21-30', '31-40', '41-45', '46-50', '51-60', '61-80']
dspat['Age_group'] = pd.cut(dspat['Age'], bins=agebins, labels=agelabel, include_lowest=True)

#Check remaining features for Null values
len(dspat[(pd.isnull(dspat['Level_of_Stress']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['Chronic_kidney_disease']) == True)])  ## This one works
len(dspat[(pd.isnull(dspat['Adrenal_and_thyroid_disorders']) == True)])  ## This one works

dstt = dspat[(pd.isnull(dspat['Genetic_Pedigree_Coefficient']) == True)]



### Analysis of Null Genetic Pedigree Coefficient records
dspat['Genetic_Pedigree_Coefficient'].hist()

dspat.plot(kind = 'scatter', x='Genetic_Pedigree_Coefficient', y='Blood_Pressure_Abnormality')
plt.show()

dstt2 = dspat.loc[ : , ['Adrenal_and_thyroid_disorders', 'Genetic_Pedigree_Coefficient']]
dstt2.boxplot(by='Adrenal_and_thyroid_disorders')

dstt2 = dspat.loc[ : , ['Level_of_Stress', 'Genetic_Pedigree_Coefficient']]
dstt2.boxplot(by='Level_of_Stress')

dstt2 = dspat.loc[ : , ['Blood_Pressure_Abnormality', 'Genetic_Pedigree_Coefficient']]
dstt2.boxplot(by='Blood_Pressure_Abnormality')

sns.boxplot(x=dspat["Genetic_Pedigree_Coefficient"])

dstt = dspat[(dspat['Adrenal_and_thyroid_disorders']==1)]
dstt['Genetic_Pedigree_Coefficient'].plot(kind = 'hist', alpha=0.5)
dstt = dspat[(dspat['Chronic_kidney_disease']==1)]
dstt['Genetic_Pedigree_Coefficient'].plot(kind = 'hist', alpha=0.5)
plt.show()

### For observations having blood pressure abnormality, thyroid and Kidney disorder = 0 set Pedigree coeff to 0.5
dspat['Genetic_Pedigree_Coefficient'].fillna(value = -1, inplace=True)
dstt = dspat[(dspat['Genetic_Pedigree_Coefficient']==-1)]
dstt2 = dstt[((dspat['Blood_Pressure_Abnormality']==1) 
& (dspat['Adrenal_and_thyroid_disorders']==1) & (dspat['Chronic_kidney_disease']==1) 
& (dspat['Genetic_Pedigree_Coefficient']==-1))]

dstt2 = dstt[((dspat['Blood_Pressure_Abnormality']==0) 
& (dspat['Adrenal_and_thyroid_disorders']==0) & (dspat['Chronic_kidney_disease']==0))]

dspat.loc[((dspat.Blood_Pressure_Abnormality == 0) 
& (dspat.Adrenal_and_thyroid_disorders == 0 )
& (dspat.Chronic_kidney_disease == 0 )
& (dspat.Genetic_Pedigree_Coefficient == -1 ))
, 'Genetic_Pedigree_Coefficient'] = 0.5

### For observations having blood pressure abnormality, thyroid and Kidney disorder = 1 set 
### Pedigree coeff to 0.2 and 0.8 for 7 observations each
dspat.loc[((dspat.Blood_Pressure_Abnormality == 1) 
& (dspat.Adrenal_and_thyroid_disorders == 1 )
& (dspat.Chronic_kidney_disease == 1 )
& (dspat.Patient_Number < 1075 )
& (dspat.Genetic_Pedigree_Coefficient == -1 )
)
, 'Genetic_Pedigree_Coefficient'] = 0.2

dspat.loc[((dspat.Blood_Pressure_Abnormality == 1) 
& (dspat.Adrenal_and_thyroid_disorders == 1 )
& (dspat.Chronic_kidney_disease == 1 )
& (dspat.Patient_Number > 1074 )
& (dspat.Genetic_Pedigree_Coefficient == -1 )
)
, 'Genetic_Pedigree_Coefficient'] = 0.8

### Remaining observations having Pedigree Coeff = -1 to be delete
dspat['Genetic_Pedigree_Coefficient'].replace(-1, np.nan, inplace = True)
dspat.dropna(subset=['Genetic_Pedigree_Coefficient'], inplace = True)

## Drop Age column as we have created Age_group column
dspat.drop('Age', axis=1, inplace=True)
## Alternatively Drop Age-group column and build the model
dspat.drop('Age_group', axis=1, inplace=True)

dspat['Age_group'].hist()

## Create Dummy Columns for Age_group
dspat = pd.concat([dspat, pd.DataFrame(pd.get_dummies(dspat.Age_group).iloc[:, 1:])], axis=1)
dspat.drop('Age_group', axis=1, inplace=True)

## Create Dummy Columns for Level_of_Stress
dspat = pd.concat([dspat, pd.DataFrame(pd.get_dummies(dspat.Level_of_Stress).iloc[:, 1:])], axis=1)
dspat.drop('Level_of_Stress', axis=1, inplace=True)


dspat.to_csv("dspat.csv", encoding='utf-8', index=False)

######################################
####### Data Cleaning complete
######################################
##### EDA Starts


dspat['Adrenal_and_thyroid_disorders'].hist()

alpha_color = .5
dspat['Blood_Pressure_Abnormality'].value_counts().plot(title='Blood_Pressure_Abnormality', kind='bar', color = ['b', 'r'], alpha = alpha_color)
dspat['Sex'].value_counts().plot(title='Sex', kind='bar', color = ['b', 'r'], alpha = alpha_color)
dspat['Pregnancy'].value_counts().plot(title='Pregnancy', kind='bar', color = ['b', 'r'], alpha = alpha_color)



dspat.groupby(['Sex'])['Smoking'].value_counts().unstack().plot(kind='bar',stacked=True)
plt.show()

dspat['Chronic_kidney_disease'].value_counts().plot(title='Chronic_kidney_disease', kind='bar', color = ['b', 'r'], alpha = alpha_color)
## Group by Sex
dspat.groupby(['Sex'])['Chronic_kidney_disease'].value_counts().unstack().plot(kind='bar',stacked=True)
plt.show()

dspat.groupby(['Sex'])['Adrenal_and_thyroid_disorders'].value_counts().unstack().plot(kind='bar',stacked=True)
plt.show()


## Boxplot for Level_of_Hemoglobin
sns.boxplot(x=dspat["Level_of_Hemoglobin"])
## Boxplot for Level_of_Hemoglobin group by Sex
dstt = dspat.loc[ : , ['Level_of_Hemoglobin', 'Sex']]
dstt.boxplot(by='Sex')

stats.probplot(dspat['Level_of_Hemoglobin'], dist="norm", plot=plt)
plt.title("Normal Q-Q plot : Level_of_Hemoglobin")
plt.show()

## Boxplot for Level_of_Hemoglobin for Female patients group by Age_group
dstt = dspat[(dspat['Sex']==1)]
dstt2 = dstt.loc[ : , ['Level_of_Hemoglobin', 'Age_group']]
dstt2.boxplot(by='Age_group')
## Analysis of Female patients wrt age_group
dstt.groupby(['Age_group'])['Pregnancy'].value_counts().unstack().plot(kind='bar',stacked=True)
plt.show()

## Analysis of BMI
#ds['total_pymnt'].value_counts().plot(kind='bar')
# Scatter plot of Age Vs BMI
dstt.plot(kind = 'scatter', x='Age', y='BMI')
plt.show()

dstt = dspat[(dspat['Sex']==1)]
dstt2 = dstt.loc[ : , ['BMI', 'Age_group']]
dstt2.boxplot(by='Age_group')

dstt = dspat.loc[ : , ['Level_of_Hemoglobin', 'Age', 'BMI', 'Physical_activity', 
                       'salt_content_in_the_diet', 'alcohol_consumption_per_day']]
scatter_matrix(dstt, alpha=0.2, figsize=(6, 6), diagonal='kde')

## Analysis of Physical Activity
dstt = dspat[(dspat['Sex']==1)]
dspat.plot(kind = 'scatter', x='Physical_activity', y='Level_of_Hemoglobin')
plt.show()

## Analysis of Level_of_Stress
dstt = dspat[(dspat['Sex']==1)]
dstt.groupby(['Age_group'])['Level_of_Stress'].value_counts().unstack().plot(kind='bar',stacked=True)
plt.show()

dspat.describe()


