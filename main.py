from fastapi import FastAPI
import models.ml.regressor as reg
from models.Property import Property
import pickle
import pandas as pd


app = FastAPI(
    title="Romanian Housing ML API",
    description="API for Romanian Housing price predictiion",
    version="1.0"
)


@app.on_event('startup')
async def load_model():
    reg.model=pickle.load(open('models/ml/xgb_reg_model.pkl', 'rb'))


@app.post('/predict', tags=["predictions"])
async def get_prediction(property: Property):
    data = dict(property)
    print(data)
    print(pd.DataFrame([data]))
    prediction = round(float(reg.model.predict(pd.DataFrame([data]))[0]), 2)
    return {"prediction": prediction}

