import pandas as pd
import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass


    def transform(self, X_df):
        X_df_new = pd.concat(
            [X_df.get(['Fare', 'Age', 'SibSp', 'Parch']),
             pd.get_dummies(X_df.Sex, prefix='Sex', drop_first=True),
             pd.get_dummies(X_df.Pclass, prefix='Pclass', drop_first=True),
             pd.get_dummies(
                 X_df.Embarked, prefix='Embarked', drop_first=True)],
            axis=1)
        
        table = pd.pivot_table(X_df_new,values='Age', index=['Parch','SibSp'], aggfunc=np.median)
# Define function to return value of this pivot_table
        truc = X_df_new[pd.isnull(X_df_new['Age'])]
        def fage(x):
            return table.loc[x['Parch'],x['SibSp']]
        
# Replace missing values
        X_df_new=X_df_new['Age'].fillna(truc.apply(fage, axis=1))   
             
        #X_df_new = X_df_new['Age'].fillna(X_df_new['Age'].mean())
        XX = X_df_new.values
        return XX

        
        