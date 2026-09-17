import os
from dataclasses import dataclass, asdict

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import joblib


@dataclass(frozen=True)
class Config:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    random_state: int
    objective: str
    enable_categorical: bool
    min_child_weight: int
    reg_lambda: float
    early_stopping_rounds: int

class CarsTrainer():

    def __init__(self,config: Config):
        self.config = config
        self.model = XGBRegressor(**asdict(self.config))
        self.predictions = {}

    def train(self,X_train,y_train,X_test,y_test_log):
        """
        Allena il modello
        :param X_train: set di train X (feature)
        :param y_train: set di train y (obiettivo)
        :param X_test:  set di test X (feature)
        :param y_test_log: set di test y (obiettivo)
        :return:
        """
        self.model.fit(X_train,y_train,eval_set=[(X_test,y_test_log)],verbose=False)

    def predict(self,X_test):
        """
        Predice il target
        :param X_test: set di test X (feature)
        :return:
        """
        self.predictions = {}
        preds = self.model.predict(X_test)
        self.predictions = preds

    def save_model(self):
        """
        Salva il modello
        :return:
        """
        os.makedirs('models',exist_ok=True)
        joblib.dump(self.model,'models/best_cars_pred.joblib')
        print('Modello salvato',self.model)

    def print_score(self,y_true,y_pred):
        '''mae = mean_absolute_error(y_true,y_pred)
        rmse = root_mean_squared_error(y_true,y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        print("*"*20 + f" MAE: {mae:.4f} " + "*"*20)
        print("*"*20 + f" RMSE: {rmse:.4f} " + "*"*20)
        print("*"*20 + f" MAPE: {mape:.2f}" + "*"*20)'''

        rmsle = root_mean_squared_error(y_true,y_pred)
        print("*"*20 + f" RMSLE: {rmsle:.4f} " + "*"*20)