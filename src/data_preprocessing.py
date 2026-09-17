import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder


class CarsProcessor():
    def __init__(self ):
        self.preprocessor = None
        self.numeric_col = None
        self.categorical_col = None


    def _feature_engineering(self,X):
        """
        Gestione feature aggiuntive
        :param X: set di partenza
        :return: set aggiornato
        """
        X_out = X.copy()
        current_year = datetime.now().year
        X_out['Year'] = pd.to_numeric(X_out['Year'],errors='coerce')
        X_out['Kilometers'] = pd.to_numeric(X_out['Kilometers'],errors='coerce')
        X_out['Old_year'] = current_year - X_out['Year']
        #X_out['Km_Per_year'] = X_out['Kilometers'] / X_out['Old_year']
        return X_out

    def fit(self,X_train,y_train):
        """
        Fit per gestire variabili categoriche
        :param X_train: set di train su X (feature)
        :param y_train: set di train su y (obiettivo)
        :return:
        """
        X_train = self._feature_engineering(X_train)

        self.numeric_col = X_train.select_dtypes(exclude=['object','category','string']).columns.tolist()
        self.categorical_col = X_train.select_dtypes(include=['object','category','string']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('target_encoder', TargetEncoder(smooth=10,random_state=42,target_type='continuous')),
        ])


        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_col),
            ('cat', categorical_transformer, self.categorical_col)
        ])

        self.preprocessor.fit(X_train,y_train)
        return self

    def transform(self,X):
        """
        Applica logica di feature engineering e pipeline sui dati di input
        :param X: set di partenza
        :return: Array Numpy con le feature aggiornate
        """
        X = self._feature_engineering(X)
        X_transformed =  self.preprocessor.transform(X)
        if hasattr(X_transformed, "values"):
            return X_transformed.values
        return np.asarray(X_transformed)

    def fit_transform(self,X_train,y_train):
        """
        Esegue il fit della pipeline e la transformazione in unico passaggio
        :param X_train: set di train su X (feature)
        :param y_train: set di train su y (obiettivo)
        :return: Array Numpy con le feature aggiornate
        """
        return self.fit(X_train,y_train).transform(X_train)

    def save_preprocessor(self):
        """
        Salva il processor relativo alla trasformazione
        :return:
        """
        os.makedirs('models',exist_ok=True)
        joblib.dump(self,'models/cars_preprocessor.joblib')
        print('Preprocessor salvato',self)