import json
import random
import requests
import time
from llms import LLMAPI, get_logger
from typing import List
from pydantic import BaseModel

logger = get_logger(__name__, 'INFO')

with open('assets/sense_keys.txt', 'r') as fr:
    keys = fr.readlines()
    ks = []
    for k in keys:
        ks.append(k.strip())

key_id = random.randint(0, len(ks)-1)
sense_key = ks[key_id]
sense_url = 'https://lm_experience.sensetime.com/test/v1/nlp/chat/completions' 
headers = {
    'Content-Type': 'application/json',
    'Authorization': sense_key
}

class SenseChatAPI(LLMAPI):
    def __init__(self, 
                 model_name='sensechat', 
                 model_path=None, 
                 model_version='default'):
        super(SenseChatAPI, self).__init__(model_name, model_path, model_version)
        self.name = 'sensechat'
        
    def generate(self, item: BaseModel) -> List[str]:
        global sense_key, sense_url, headers
        prompt = item.prompt
        
        if not prompt:
            return []

        if type(prompt) is not list:
            prompt = [prompt]

        error = None
        num_failures = 0
        while num_failures < 5:
            try:
                sense_replies = []
                for p in prompt:
                    chat_instance = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]

                    sense_data = dict(
                        messages=chat_instance,
                        temperature=item.temperature,
                        top_p=item.top_p,
                        max_new_tokens=item.max_new_tokens,
                        repetition_penalty=1, 
                        user="test"
                    )
                    
                    response = requests.post(sense_url, headers=headers, json=sense_data)
                    sense_reply = json.loads(response.text)['data']['choices'][0]['message']

                    sense_replies.append(sense_reply)
                
                return sense_replies
            except requests.ConnectionError as error:
                logger.warning(error)
                logger.warning(f"Reach Rate Limit!")
                num_failures += 1
                time.sleep(5)
            except Exception as error:
                logger.warning(error)
                num_failures += 1
                time.sleep(5)

        raise error
    
if __name__ == '__main__':
    model_api = SenseChatAPI()
