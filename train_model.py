import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data_preprocessing import CarsProcessor
from src.model import CarsTrainer, Config
import numpy as np


def _read_yaml(config_path: str):
    with open(config_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    return yaml_content


def test():
    """
    Esegue il processo di training
    :return:
    """
    cur_df = pd.read_csv('data/car_prices.csv')
    cur_df = cur_df.dropna(subset=['Price'])

    X = cur_df.drop(columns=['Price','Seller','Body_Type','Type','Seats','Doors','Cylinders','Upholstery'])
    y= cur_df['Price']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    preprocessor = CarsProcessor()
    X_train_transformed = preprocessor.fit_transform(X_train,y_train_log)
    X_test_transformed = preprocessor.transform(X_test)


    yaml_content = _read_yaml(config_path='src/config_param_model.yaml')
    config = Config(**yaml_content['model_hyperparameters'])

    print("*"*20 + " Inizio Training " + "*"*20)
    trainer = CarsTrainer(config)
    trainer.train(X_train_transformed,y_train_log,X_test_transformed,y_test_log)
    print("*"*20 + " Fine Training " + "*"*20)
    trainer.predict(X_test_transformed)

    y_preds = trainer.predictions
    preds_real = np.expm1(y_preds)
    trainer.print_score(y_test_log,y_preds)

    importanze = trainer.model.get_booster().get_score(importance_type='gain')

    df_importanze = pd.DataFrame({
        'Feature': importanze.keys(),
        'Importanza (Gain)': importanze.values()
    }).sort_values(by='Importanza (Gain)', ascending=False)

    print(df_importanze)

    preprocessor.save_preprocessor()
    trainer.save_model()

if __name__ == '__main__':
    test()