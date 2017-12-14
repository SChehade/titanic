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
            
        
        new_age=X_df_new.groupby(["Parch", "SibSp"])["Age"].transform("median")
        X_df_new['Age']=new_age
            
        X_df_new = X_df_new.fillna(-1)
      
        
        XX = X_df_new.values
        return XX

        
        