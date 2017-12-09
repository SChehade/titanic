# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:58:20 2017

@author: SC249077
"""

import os
import glob
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd

#loading the data
train_filename = 'data/train.csv'
data = pd.read_csv(train_filename)
y_df = data['Survived']
X_df = data.drop(['Survived', 'PassengerId'], axis=1)
print("Head")
print(X_df.head(5))
print("data count")
print(data.count())
print("data describe")
print(data.describe())
print("number survived")
print(data.groupby('Survived').count())
#print("frequency count")
#print(data['Sex'].value_counts())

# Tests for classification
temp1 = data['Sex'].value_counts()
temp2 = pd.pivot_table(data,values='Survived',index=['Sex'])
print('Frequency Table for Gender:' )
print(temp1)

print('\n Probility of Surviving for each Gender:' )
print(temp2)


temp1 = data['Pclass'].value_counts()
temp2 = pd.pivot_table(data,values='Survived',index=['Pclass'])
print('Frequency Table for Pclass:' )
print(temp1)

print('\n Probility of Surviving for each Class:' )
print(temp2)


temp1 = data['SibSp'].value_counts()
temp2 = pd.pivot_table(data,values='Survived',index=['SibSp'])
print('Frequency Table for Number of siblings/spouses:' )
print(temp1)

print('\n Probility of Surviving for Number of siblings/spouses:' )
print(temp2)


temp1 = data['Parch'].value_counts()
temp2 = pd.pivot_table(data,values='Survived',index=['Parch'])
print('Frequency Table for Number of parents/children:' )
print(temp1)

print('\n Probility of Surviving for Number of parents/children:' )
print(temp2)

#Missing data
print('How much missing data')
print(data.apply(lambda x: sum(x.isnull()),axis=0))

#some plots

#scatter plots
from pandas.plotting import scatter_matrix
scatter_matrix(data.get(['Fare', 'Pclass', 'Age']), alpha=0.2,
               figsize=(8, 8), diagonal='kde');
               
data_plot = data.get(['Age', 'Survived'])
data_plot = data.assign(LogFare=lambda x : np.log(x.Fare + 10.))
scatter_matrix(data_plot.get(['Age', 'LogFare']), alpha=0.2, figsize=(8, 8), diagonal='kde');

#log for the fare plot
data_plot.plot(kind='scatter', x='Age', y='LogFare', c='Survived', s=50, cmap=plt.cm.Paired);


#another visualisation
import seaborn as sns

sns.set()
sns.set_style("whitegrid")
sns.jointplot(data_plot.Age[data_plot.Survived == 1],
              data_plot.LogFare[data_plot.Survived == 1],
              kind="kde", size=7, space=0, color="b");

sns.jointplot(data_plot.Age[data_plot.Survived == 0],
              data_plot.LogFare[data_plot.Survived == 0],
              kind="kde", size=7, space=0, color="y");