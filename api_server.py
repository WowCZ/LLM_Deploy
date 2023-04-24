import uvicorn
from fastapi import FastAPI # for local api
from llm_api import * # ChatGLMAPI, T5API, DavinciAPI, TurboAPI, BloomAPI
import socket
from typing import Optional, Dict, Union
from pydantic import BaseModel
import logging
import os
import argparse
import random
from flask import Flask, request
from waitress import serve

parser = argparse.ArgumentParser(description='llm api server')
parser.add_argument('--api', type=str, default='ChatGLMAPI', help='Supported API: [ChatGLMAPI, T5API, DavinciAPI, TurboAPI, BloomAPI]')
parser.add_argument('--server', type=str, default='Flask', help='Supported Server: [Flask, FastAPI]')
args = parser.parse_args()

logger = logging.getLogger(__name__)


def isInuseLinux(port):
    #lsof -i:8080
    #not show pid to avoid complex
    if os.popen('netstat -na | grep :' + str(port)).readlines():
        portIsUse = True
        print('%d is inuse' % port)
    else:
        portIsUse = False
        print('%d is free' % port)
    return portIsUse

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
        return ip

class Item(BaseModel):
    prompt: Union[str, list]
    temperature: float=0.95
    max_new_tokens: int=2048
    top_p: float=0.7
    num_return: int=1
    do_sample: bool=True
    seed: Optional[int]=None

model_api = eval(args.api)()

if args.server == 'Flask':
    ### Flask Server
    app = Flask(__name__)
    @app.route('/generate', methods=['POST'])
    def generate() -> Dict:
        # item = json.dumps(json.loads(request.data), ensure_ascii=False)
        item = Item.parse_raw(request.data)
        try:
            output = model_api.generate(item)
            output = [output] if output is str else output
            return {"outputs": output}
        except:
            output = item.prompt
            return {"outputs": output}
else:
    ### FastAPI Server
    app = FastAPI()
    @app.post('/generate')
    async def generate(item: Item) -> Dict:
        try:
            output = model_api.generate(item)
            output = [output] if output is str else output
            return {"outputs": output}
        except:
            output = item.prompt
            return {"outputs": output}


if __name__ == '__main__':
    host_ip = get_host_ip()
    port = random.randint(5000, 10000)
    while isInuseLinux(port):
        port = random.randint(5000, 10000)

    server_info_record = 'server_info_record.txt'
    fw = open(server_info_record, 'a')
    server_url = f'http://{host_ip}:{port}/generate\n'
    fw.write(server_url)
    fw.flush()
    fw.close()

    print("api IP = Host:Port = ", host_ip,":",port)
    logger.info("api IP = Host:Port = ", host_ip,":",port)
    if args.server == 'Flask':
        ### Flask Server
        app.config['JSON_AS_ASCII'] = False
        serve(app, host='0.0.0.0', port=port)
    else:
        ### FastAPI Server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    # curl -H "Content-Type: application/json" -X POST http://10.140.24.72:5001/generate -d "@cn_gen.json"