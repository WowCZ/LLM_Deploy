import os
import argparse
import random
import uvicorn
import socket
from llm_api import *
from llm_api import get_logger
from waitress import serve
from fastapi import FastAPI
from pydantic import BaseModel
from flask import Flask, request
from typing import Optional, Dict, Union

parser = argparse.ArgumentParser(description='llm api server')
parser.add_argument('--api', type=str, default='ChatGLMAPI', help='Supported API: [ChatGLMAPI, T5API, DavinciAPI, TurboAPI, BloomAPI]')
parser.add_argument('--server', type=str, default='Flask', help='Supported Server: [Flask, FastAPI]')
args = parser.parse_args()

logger = get_logger(__name__, 'INFO')

def isInuseLinux(port):
    #lsof -i:8080
    #not show pid to avoid complex
    if os.popen('netstat -na | grep :' + str(port)).readlines():
        portIsUse = True
        logger.info('%d is inuse' % port)
    else:
        portIsUse = False
        logger.info('%d is free' % port)
    return portIsUse

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
        return ip

class GenItem(BaseModel):
    prompt: Union[str, list]
    temperature: float=0.0
    max_new_tokens: int=1024
    top_p: float=1.0
    num_return: int=1
    do_sample: bool=False
    seed: Optional[int]=None

class ScoreItem(BaseModel):
    prompt: Union[str, list]
    target: Union[str, list]

model_api = eval(args.api)()

if args.server == 'Flask':
    ### Flask Server
    app = Flask(__name__)
    @app.route('/generate', methods=['POST'])
    def generate() -> Dict:
        item = GenItem.parse_raw(request.data)
        output = model_api.generate(item)
        output = [output] if output is str else output
        return {"output": output}
    
    if 'score' in model_api.supported_types:
        @app.route('/score', methods=['POST'])
        def score() -> Dict:
            item = ScoreItem.parse_raw(request.data)
            output = model_api.score(item)
            return {"log_prob": output}
    
else:
    ### FastAPI Server
    app = FastAPI()
    @app.post('/generate')
    async def generate(item: GenItem) -> Dict:
        try:
            output = model_api.generate(item)
            output = [output] if output is str else output
            return {"output": output}
        except:
            output = item.prompt
            return {"output": output}
    
    if 'score' in model_api.supported_types:
        @app.post('/generate')
        def score(item: ScoreItem) -> Dict:
            output = model_api.score(item)
            return {"log_prob": output}


if __name__ == '__main__':
    host_ip = get_host_ip()
    port = random.randint(5000, 10000)
    while isInuseLinux(port):
        port = random.randint(5000, 10000)

    llm_name = model_api.name
    server_info_record = f'copywriting/urls/{llm_name}_server_info.txt'

    fw = open(server_info_record, 'a')
    server_url = f'http://{host_ip}:{port}/generate\n'
    fw.write(server_url)
    if 'score' in model_api.supported_types:
        server_url = f'http://{host_ip}:{port}/score\n'
        fw.write(server_url)
    
    fw.flush()
    fw.close()

    logger.info(f'#{llm_name.upper()}# has been deployed, API INFO as below:')
    logger.info(f"API IP = Host:Port = {host_ip}:{port}")
    if args.server == 'Flask':
        ### Flask Server
        app.config['JSON_AS_ASCII'] = False
        serve(app, host='0.0.0.0', port=port)
    else:
        ### FastAPI Server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    # curl -H "Content-Type: application/json" -X POST http://10.140.24.50:7396/generate -d "@test/cn_gen.json"
    # curl -H "Content-Type: application/json" -X POST http://10.140.24.61:8454/score -d "@test/cn_score.json"