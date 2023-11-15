"""Code to active API"""

import pickle
from fastapi import FastAPI, Body
import uvicorn
import pandas as pd
import joblib


# Create a FastAPI instance
app = FastAPI(debug=True)

# Import serialized data

#model = pickle.load(open('model_met1.sav', 'rb'))
pipeline = joblib.load('./ressources/model.joblib')

# Endpoints
@app.get('/')
def home():
    """
    Welcome message.
    Args:  
    - None.  
    Returns:  
    - Message (string).  
    """
    return 'Hello, my API works!!'

@app.get('/prediction/')
def get_prediction(json_client: dict = Body({})):
    """
    Calculates the probability of default for a client.  
    Args:  
    - client data (json).  
    Returns:    
    - probability of default (dict).
    """
    df_one_client = pd.Series(json_client).to_frame().transpose()
    probability = pipeline[1].predict_proba(df_one_client)[:, 1][0]
    return {'probability': probability}