import time
import openai
from llms import LLMAPI
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

openai_key = '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'

class GPT4API(LLMAPI):
    def __init__(self, 
                 model_name='gpt-4', 
                 model_path=None,
                 model_version='default'):
        super(GPT4API, self).__init__(model_name, model_path, model_version)
        self.name = 'gpt4'
        
    def generate(self, item: BaseModel) -> List[str]:
        openai.api_key = openai_key
        openai.organization = 'org-Ocn8x8Go7Sh57REO2KrpCDpr'

        prompt = item.prompt
        if not prompt:
            return []

        if type(prompt) is not list:
            prompt = [prompt]

        error = None
        num_failures = 0
        while num_failures < 5:
            try:
                gpt4_replies = []
                for p in prompt:
                    chat_instance = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": p}]

                    completion = openai.ChatCompletion.create(model="gpt-4",
                                                                messages=chat_instance,
                                                                n=item.num_return)
                    gpt4_reply = completion.choices[0].message.content.strip()
                    gpt4_replies.append(gpt4_reply)

                return gpt4_replies
            except openai.error.RateLimitError as error:
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
    model_api = GPT4API()
