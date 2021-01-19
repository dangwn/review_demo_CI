'''
Author: Dan Gawne
Date: 2021-01-04
'''

#%%
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import json
import mlflow
import pandas as pd
import waitress
import numpy as np

import imdb_utils

from flask import Flask, request

#%%
#-----------------------------------------------------------------------------
# Get YAML Paths
#-----------------------------------------------------------------------------
import yaml

with open('file_paths.yaml', 'r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

#%%
#-----------------------------------------------------------------------------
# Which model to use
#-----------------------------------------------------------------------------
random_forest = False
if random_forest:
    experiment_name = 'Reviews - Random Forest'
else:
    experiment_name = 'Reviews - Logistic Regression'

tracking_uri = r'file:' + file_paths['model_loc'][0]
vector_path = file_paths['vector_headers'][0]

#%%
#-----------------------------------------------------------------------------
# Model
#-----------------------------------------------------------------------------
model = mlflow.sklearn.load_model(tracking_uri)


#%%
#-----------------------------------------------------------------------------
# Flask App
#-----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return 'Welcome to this review API'

@app.route('/invocations', methods = ['POST'])
def predict():
    '''
    Data must come in as {'review', review}
    '''
    review = json.loads(request.data)['review']
    review_vector = imdb_utils.vectorize_doc(review, vector_path)
    prediction = model.predict(np.array(review_vector).reshape(1,-1))[0]
    if prediction:
        return f'Postive'
    else:
        return f'Negative'
    
    

#%%
#-----------------------------------------------------------------------------
# Start Server
#-----------------------------------------------------------------------------
waitress.serve(app, host = '0.0.0.0', port = 6000)
