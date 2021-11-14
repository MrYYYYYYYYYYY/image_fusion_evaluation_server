import json
import time
import requests
import yaml
import os
from flask import Flask, request, render_template, send_file, abort, jsonify
import evalutaor
import base64
import numpy as np
import cv2
from  skimage.io import imread
import pdb
from log import Logger, LoggerF
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor

with open(os.path.join(os.path.split(__file__)[0], 'config.yaml')) as f:
    config = yaml.load(f)

app = Flask(__name__)

logger = Logger()
logger.init_app(app)
log = LoggerF('./logs/error.log',level = 'debug')
executor = ThreadPoolExecutor(2)

@app.route('/evaluate',methods=['POST'])
def main():

    state_code = 200
    data = request.get_data()
    data = json.loads(data.decode())
    missing_key = set(['submitId','fused_result', 'dataset', 'task']) - set(data.keys())
    if(len(missing_key)==0):
        executor.submit(evaluation, data)
        info = 'start evaluation'
    else:
        info = "missing keys:" + ".".join(missing_key)
        state_code = 400

    result = {}
    result['code'] = state_code
    result['info'] = info

    return result

def evaluation(data):
    result = {}
    task = data['task']
    submitId = data['submitId']
    dataset = data['dataset']
    log.logger.info("start evaluation submitId " + str(submitId))
    try:
        eva = evalutaor.Evaluator()
        data_conf = config['task'][task]['dataset'][dataset]
        imgs = data_conf['double']['data']
        suffix = data_conf['double']['suffix']
        for img_name in imgs:
            fused_result = requests.get(data['fused_result'][img_name]).content
            #fused_result = base64.b64decode(data['fused_result'][img_name].encode())
            fused = np.frombuffer(fused_result,np.uint8)
            fused = cv2.imdecode(fused,cv2.IMREAD_COLOR)
            imgA = imread(os.path.join(data_conf['root'],'double',img_name + suffix[0]))
            imgB = imread(os.path.join(data_conf['root'],'double',img_name + suffix[1]))
            result[img_name] = eva.get_evaluation(imgA, imgB, fused, metrics=config['task'][task]['metrics'])
            #result[img_name] = {'qg': 0.01,"qmi":0.5}

    except Exception as e:
        log.logger.error(traceback.format_exc())
        result['result'] = 'evalution error'

    call_back(result, submitId)

def call_back(eva_result, submitId):
    log.logger.info("start call back submitId " + str(submitId))
    call_back_url = config['call_back_url']
    result = {}
    result['evaluateMap'] = eva_result
    result["submitId"] = submitId
    result = requests.post(call_back_url, json.dumps(eva_result))
    log.logger.info(result.text)






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12000, debug = True)

