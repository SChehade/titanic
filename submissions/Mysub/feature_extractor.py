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
        table.rename(columns={'Age': 'Age_median'}, inplace=True)
        # Define function to return value of this pivot_table
        #truc = X_df_new[pd.isnull(X_df_new['Age'])]
        #def fage(x):
         #   return table.loc[x['Parch'],x['SibSp']]
        #X_df_new=X_df_new.fillna(12) 

        table.reset_index(inplace=True)
        X_df_new=pd.merge(X_df_new,table,on=['Parch','SibSp'])
        X_df_new.loc[pd.isnull(X_df_new['Age']),'Age'] = X_df_new['Age_median']
# Replace missing values
          
             
        #X_df_new = X_df_new['Age'].fillna(X_df_new['Age'].mean())
        X_df_new.drop('Age_median',axis=1,inplace=True)
        X_df_new = X_df_new.fillna(X_df_new.mean())
        #X_df_new=X_df_new.fillna(-1) 
        
        XX = X_df_new.values
        return XX

        
        