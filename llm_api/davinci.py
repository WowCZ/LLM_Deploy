import os
import openai
from llm_api import LLMAPI
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

openai_key = os.environ['OPENAI_KEY'] if 'OPENAI_KEY' in os.environ else None
# assert openai_key, "No OpenAI API key detected in the environment"

params = {"temperature": 1.0, "top_p": 1.0, "num_generations": 1, "max_tokens": 512}

class DavinciAPI(LLMAPI):
    def __init__(self, model_name='text-davinci-003', model_path=None):
        super(DavinciAPI, self).__init__(model_name, model_path)
        
    def generate(self, item: BaseModel) -> List[str]:
        openai.api_key = openai_key

        prompt = item.prompt
        if type(prompt) is not list:
            prompt = [prompt]
        
        davinci_replies = []
        for p in prompt:
            completion = openai.Completion.create(model="text-davinci-003",
                                                    prompt=p,
                                                    temperature=item.temperature,
                                                    top_p=item.top_p,
                                                    max_tokens=item.max_new_tokens,
                                                    n=item.num_return)
            davinci_reply = completion.choices[0].text.strip()
            davinci_replies.append(davinci_reply)
        
        return davinci_replies
    
if __name__ == '__main__':
    model_api = DavinciAPI()
