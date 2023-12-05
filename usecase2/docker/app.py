import pandas as pd
from flask import Flask, jsonify,request
import os
from fastai.collab import CollabDataLoaders, collab_learner
from fastai.learner import load_learner
import torch
import numpy as np
from google.cloud import storage

import warnings

print('AIP_STORAGE_URI', os.environ['AIP_STORAGE_URI'])

dst_file_name = '/app/learner.pth'
model_path = os.environ['AIP_STORAGE_URI']
bucket_name = model_path.split('//')[1].split('/')[0]
prefix = "/".join(model_path.split('//')[1].split('/')[1:])

storage_client = storage.Client()
blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

for blob in blobs:
    print('blob', blob.name)
    if blob.name.endswith('/model'):
        blob.download_to_filename(dst_file_name)

app = Flask(__name__)

learner = load_learner(dst_file_name)

@app.route('/predict',methods=['POST','GET'])
def predict():
    output = "default"
    try:
        req = request.json.get('instances')
        df = pd.DataFrame({'user':[a[0] for a in req], 'item':[a[1] for a in req]})
        dl = learner.dls.test_dl(df, with_labels=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aaa = learner.get_preds(dl=dl)        
            prediction = [x.tolist()[0] for x in aaa[0]]

        #postprocessing
        output = {'instance':req, 'predictions':prediction}
    except Exception as error:
        output = {"error": f"{error}", "message": msg}
    finally:
        return jsonify(output)

@app.route('/healthz')
def healthz():
    return "OK"


if __name__=='__main__':
    app.run(host='0.0.0.0')
