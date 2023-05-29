import os
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

def api_server(api_name: str, server_wrapper: str, url_save_path: str, gen_config: dict):
    class GenItem(BaseModel):
        prompt: Union[str, list]
        temperature: float=gen_config['temperature']
        max_new_tokens: int=gen_config['max_new_tokens']
        top_p: float=gen_config['top_p']
        num_return: int=gen_config['num_return']
        do_sample: bool=gen_config['do_sample']
        seed: Optional[int]=gen_config['seed']

    class ScoreItem(BaseModel):
        prompt: Union[str, list]
        target: Union[str, list]

    model_api = eval(api_name)()

    if server_wrapper == 'Flask':
        ### Flask Server
        app = Flask(__name__)
        @app.route('/generate', methods=['POST'])
        def generate() -> Dict:
            item = GenItem.parse_raw(request.data)
            output = model_api.generate(item)
            output = [output] if output is str else output
            return {"outputs": output}
        
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
                return {"outputs": output}
            except:
                output = item.prompt
                return {"outputs": output}
        
        if 'score' in model_api.supported_types:
            @app.post('/generate')
            def score(item: ScoreItem) -> Dict:
                output = model_api.score(item)
                return {"log_prob": output}
            
    
    host_ip = get_host_ip()
    port = random.randint(5000, 10000)
    while isInuseLinux(port):
        port = random.randint(5000, 10000)

    if not os.path.exists(url_save_path):
        os.makedirs(url_save_path)
        
    llm_name = model_api.name
    server_info_record = f'{url_save_path}/{llm_name}_server_info.txt'

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
    if server_wrapper == 'Flask':
        ### Flask Server
        app.config['JSON_AS_ASCII'] = False
        serve(app, host='0.0.0.0', port=port)
    else:
        ### FastAPI Server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# curl -H "Content-Type: application/json" -X POST http://10.140.24.46:8610/generate -d "@examples/cn_gen.json"
# curl -H "Content-Type: application/json" -X POST http://10.140.24.61:8454/score -d "@examples/cn_score.json"