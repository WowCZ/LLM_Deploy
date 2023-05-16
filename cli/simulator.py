import json
import socket
import uvicorn
from typing import Dict
from waitress import serve
from fastapi import FastAPI
from pydantic import BaseModel
from llm_api import get_logger
from flask import Flask, request

logger = get_logger(__name__, level='INFO')

def api_simulator(simulate_task: str, llm_name: str, port: int, wrapper: str, url_path: str, inference_path: str):

    prompt_file = f'{inference_path}/{simulate_task}_{llm_name}.json'

    server_prompt_template = dict()
    simulate_generator = dict()
    with open(prompt_file, 'r') as fr:
        prompt_mapping = json.load(fr)
        for i, p in enumerate(prompt_mapping):
            prompt = p['prompt']
            model_output = p[f'{llm_name}_output']
            simulate_generator[prompt] = model_output
            server_prompt_template[f'prompt{i+1}'] = prompt

    output_prompt_file = prompt_file.replace('.json', '_prompt.json')
    with open(output_prompt_file, 'w') as fw:
        json.dump(server_prompt_template, fw, indent=4, ensure_ascii=False)

    class Item(BaseModel):
        prompt: str
        sessionStatus: int = 1
        sessionId: str = None

    if wrapper == 'Flask':
        ### Flask Server
        app = Flask(__name__)
        @app.route('/generate', methods=['POST'])
        def generate() -> Dict:
            item = Item.parse_raw(request.data)
            if item.prompt in simulate_generator:
                output = simulate_generator[item.prompt]
            else:
                print(f'>>> Bad Prompt {item.prompt}!')
                output = item.prompt
            return {"outputs": [{"modelId": llm_name, "output": output}]}
    else:
        ### FastAPI Server
        app = FastAPI()
        @app.post('/generate')
        async def generate(item: Item) -> Dict:
            if item.prompt in simulate_generator:
                output = simulate_generator[item.prompt]
            else:
                print(f'>>> Bad Prompt {item.prompt}!')
                output = item.prompt
            return {"outputs": [{"modelId": llm_name, "output": output}]}

    host_ip = socket.gethostbyname(socket.gethostname())
    server_info_record = f'{url_path}/{llm_name}_simulator.txt'

    fw = open(server_info_record, 'a')
    server_url = f'For \'{simulate_task}\' task: http://{host_ip}:{port}/generate\n'
    fw.write(server_url)
    fw.flush()
    fw.close()

    logger.info(f"API IP = Host:Port = {host_ip}:{port}")
    if wrapper == 'Flask':
        ### Flask Server
        app.config['JSON_AS_ASCII'] = False
        serve(app, host='0.0.0.0', port=port)
    else:
        ### FastAPI Server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    # curl -H "Content-Type: application/json" -X POST http://10.140.24.31:6566/generate -d "@test/sim_gen.json"