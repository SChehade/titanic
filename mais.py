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

train_filename = 'data/train.csv'
data = pd.read_csv(train_filename)
y_df = data['Survived']
X_df = data.drop(['Survived', 'PassengerId'], axis=1)
print(X_df.head(5))

data.groupby('Survived').count()