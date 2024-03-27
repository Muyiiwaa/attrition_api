from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import pickle
import numpy as np

app = FastAPI()

class Order(str, Enum):
    rice = 'rice'
    beans = 'beans'
    meat = 'meat'
    
class Features(BaseModel):
    satisfaction_level: float
    last_evaluation:float
    number_project:float
    average_montly_hours:float
    time_spend_company:float
    Work_accident:float
    promotion_last_5years:float
    dept: int
    salary: int
    

@app.post('/loan/')
async def get_interest(name: str, principal: int, time : int):
    rate = 5
    interest = (principal * rate * time)/100
    return {f"Hi {name} your Interest is {interest} and total payment is {principal + interest}"}

with open('emp_model', 'rb') as f:
    model = pickle.load(f)
 
@app.post('/order/{order}')
async def get_order(item: Features):
    """_summary_

    Args:
        item (Order): _description_
    """
    amount = (item.rice * 200) + (item.beans * 100) + (item.meat * 500) + item.tip
    return {f"Hi {item.name} your charges is ${amount}, thanks for the ${item.tip} tip"}


@app.post('/predict/')
async def predict(item: Features):
    data = dict(item)
    data = {k:[v] for k,v in data.items()}
    df = pd.DataFrame(data)
    pred = model.predict(df)
    pred = ['will churn' if x == 0 else "wont churn" for x in pred]
    return {"prediction": str(pred[0])}



