import uvicorn
from fastapi import FastAPI # for local api
from llm_api import ChatGLMAPI, T5API, DavinciAPI, TurboAPI, BloomAPI
import socket
from typing import Optional, Dict
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


model_api = BloomAPI()
# output = model_api.generate(['有这样一个故事：“我：“爸在干嘛呢？最近家里生意还好吧。”爸：“已汇””，请问这个故事的笑点在哪儿？'])
# print(output)

class Item(BaseModel):
    prompt: str
    temperature: float=0.0
    max_new_tokens: int=1024
    top_p: float=1.0
    num_return: int=1
    do_sample: bool=False
    seed: Optional[int]=None

app = FastAPI()

@app.post('/generate')
async def generate(item: Item) -> Dict:
    output = model_api.generate(item.prompt)
    return {"outputs": [output]}


host_ip = socket.gethostbyname(socket.gethostname())
port=6004
print("api IP = Host:Port = ", host_ip,":",port)
logger.info("api IP = Host:Port = ", host_ip,":",port)
uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# curl -H "Content-Type: application/json" -X POST http://10.140.24.72:5001/generate -d "@cn_gen.json"