from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel

# Бизнес логика
from joblib import load
import pandas as pd
from prep_transformer import PrepDataTransformer

model_file = 'logreg_pipe.pkl'
model = load(model_file)


# FastAPI
app = FastAPI()

class ModelItem(BaseModel):
    age:        int
    sex:        str
    cp:         str
    trestbps:   Union[float, None] = None
    chol:       Union[float, None] = None
    fbs:        Union[bool, None] = None
    restecg:    Union[str, None] = None
    thalch:     Union[float, None] = None
    exang:      Union[bool, None] = None
    oldpeak:    Union[float, None] = None
    slope:      Union[str, None] = None
    ca:         Union[float, None] = None
    thal:       Union[str, None] = None

    def get_item_series(self) -> str:
        item_data = {}
        item_data['age'] =  self.age
        item_data['sex'] =  self.sex
        item_data['cp'] =  self.cp
        item_data['trestbps'] =  self.trestbps
        item_data['chol'] =  self.chol
        item_data['fbs'] =  self.fbs
        item_data['restecg'] =  self.restecg
        item_data['thalch'] =  self.thalch
        item_data['exang'] =  self.exang
        item_data['oldpeak'] =  self.oldpeak
        item_data['slope'] =  self.slope
        item_data['ca'] =  self.ca
        item_data['thal'] =  self.thal
        out_data = pd.Series(item_data)
        return out_data


def get_prediction(item: ModelItem) -> float:
    x = item.get_item_series()
    proba = model.predict_proba(x)[0][1]
    return proba


@app.get("/")
def read_root():
    info_text = " http://localhost:8000/docs to get more API info"
    return {"Info": info_text}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str):
#     return {"item": proba_test[10], "q": q}


@app.put("/predict/{item_id}")
def update_item(item_id: int, item: ModelItem):
    return {"item_id": item_id, "item_predict": get_prediction(item)}

