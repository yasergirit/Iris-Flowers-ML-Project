from joblib import load
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np

filename = "decisiontree.joblib"

modelUploaded = load(filename)


app = FastAPI()

templates = Jinja2Templates(directory="templates")

labelsNames = ['Setosa', 'Versicolor', 'Virginica']


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})



@app.get("/predict")
async def make_prediction(request: Request, L1:float, W1:float, L2:float, W2:float):
    
    test = np.array([L1, W1, L2, W2]).reshape(1,4)
    probabilities = modelUploaded.predict_proba(test)[0]
    predicted = np.argmax(probabilities)
    
    probability = probabilities[predicted]
    
    predicted = labelsNames[predicted]
    
    return templates.TemplateResponse("prediction.html", {"request": request, 
                                                          "probabilities":probabilities,
                                                          "predicted":predicted,
                                                          "probability":probability})

