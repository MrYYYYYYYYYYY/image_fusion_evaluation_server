import requests
import json
from socket import *
from PIL import Image
import cv2
import  base64
import os
def clas(*kwargs):
    print(kwargs['img1'],kwargs['img2'],kwargs['fused'])
if __name__ == "__main__":

    img_a_binary = {}
        #img_b = open('../dataset/multi-focus/lytro/double/'+i+'-B.jpg','rb')
        #img_b_binary = base64.b64encode(img_b.read())

    data = {
        'submitId' : "0",
        'task': 'multi-focus',
        'dataset': 'lytro',
        'fused_result':  {
            'lytro-001': "https://ss2.baidu.com/-vo3dSag_xI4khGko9WTAnF6hhy/exp/w=500/sign=1f3bd9a164d0f703e6b295dc38fb5148/d52a2834349b033bce9d736c13ce36d3d539bd6b.jpg",

        },

    }

    res = requests.post('http://39.106.75.177:12000/evaluate', json.dumps(data))
    #res = requests.post('http://localhost:12000/evaluate', json.dumps(data))

    print(res.text, res.status_code)

