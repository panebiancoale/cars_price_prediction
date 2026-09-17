import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictionRequest


def _read_yaml(config_path: str):
    """
    Legge il file di configurazione per i parametri
    :param config_path:
    :return:
    """
    with open(config_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    return yaml_content

app = FastAPI(title='Cars Prediction Price API')

print('+'*5 + ' CARICAMENTO MODELLO '+ '+'*5)
try:
    preprocessor = joblib.load('models/cars_preprocessor.joblib')
    model_puro = joblib.load('models/best_cars_pred.joblib')
    print('+'*5 + ' Modello caricato '+ '+'*5)
except Exception as e:
    print(f'Errore nel caricare il modello : {str(e)}')
    model = None

@app.get("/")
def read_root():
    return {'Status':'OK','Message':'API ATTIVA'}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Gestione richiesta di predict
    :param request: richiesta con le feature inserite lato gui
    :return: prediction
    """
    if model_puro is None:
        raise HTTPException(status_code=404, detail='Modello non trovato')

    try:
        input_data = [item.model_dump() for item in request.data]
        request_df = pd.DataFrame(input_data)


        request_tranformed = preprocessor.transform(request_df)
        print(request_tranformed)

        preds_log = model_puro.predict(request_tranformed)
        preds = np.expm1(preds_log)

        return {'predictions': preds.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Errore nella predizione {str(e)}')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", reload=True)
