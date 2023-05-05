import os
import openai
from llm_api import LLMAPI
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

openai_key = os.environ['OPENAI_KEY'] if 'OPENAI_KEY' in os.environ else None
# assert openai_key, "No OpenAI API key detected in the environment"
openai.api_key = openai_key

params = {"temperature": 0.0, "top_p": 1.0, "num_generations": 1, "max_tokens": 256}

class TurboAPI(LLMAPI):
    def __init__(self, model_name='gpt-3.5-turbo', model_path=None):
        super(TurboAPI, self).__init__(model_name, model_path)
        self.name = 'turbo'
        
    def generate(self, item: BaseModel) -> List[str]:
        openai.api_key = openai_key

        prompt = item.prompt
        
        if not prompt:
            return []

        if type(prompt) is not list:
            prompt = [prompt]
        
        turbo_replies = []
        for p in prompt:
            chat_instance = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]

            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                        messages=chat_instance,
                                                        temperature=item.temperature,
                                                        top_p=item.top_p,
                                                        max_tokens=item.max_new_tokens,
                                                        n=item.num_return)
            turbo_reply = completion.choices[0].message.content.strip()
            turbo_replies.append(turbo_reply)

        return turbo_replies
    
if __name__ == '__main__':
    model_api = TurboAPI()
