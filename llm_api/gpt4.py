import os
import openai
from llm_api import LLMAPI
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

openai_key = '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'

params = {"temperature": 0.0, "top_p": 1.0, "num_generations": 1, "max_tokens": 512}

class GPT4API(LLMAPI):
    def __init__(self, model_name='gpt-4', model_path=None):
        super(GPT4API, self).__init__(model_name, model_path)
        self.name = 'gpt4'
        
    def generate(self, item: BaseModel) -> List[str]:
        openai.api_key = openai_key

        prompt = item.prompt
        if not prompt:
            return []

        if type(prompt) is not list:
            prompt = [prompt]
        
        gpt4_replies = []
        for p in prompt:
            chat_instance = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]

            completion = openai.ChatCompletion.create(model="gpt-4",
                                                        messages=chat_instance,
                                                        temperature=item.temperature,
                                                        top_p=item.top_p,
                                                        max_tokens=item.max_new_tokens,
                                                        n=item.num_return)
            gpt4_reply = completion.choices[0].message.content.strip()
            gpt4_replies.append(gpt4_reply)

        return gpt4_replies
    
if __name__ == '__main__':
    model_api = GPT4API()
