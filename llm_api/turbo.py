import os
import openai
from llm_api import LLMAPI
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

openai_key = os.environ['OPENAI_KEY'] if 'OPENAI_KEY' in os.environ else None
# assert openai_key, "No OpenAI API key detected in the environment"
openai.api_key = openai_key

params = {"temperature": 0.0, "top_p": 1.0, "num_generations": 1, "max_tokens": 1024}

class TurboAPI(LLMAPI):
    def __init__(self, model_name='gpt-3.5-turbo', model_path=None):
        super(TurboAPI, self).__init__(model_name, model_path)
    
    def _initialize_llm(self):
        tokenizer = None
        model = None

        # Test Case
        example_prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "中国有多少省份？"}]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=example_prompt,
                                                  temperature=params["temperature"],
                                                  top_p=params["top_p"],
                                                  max_tokens=params["max_tokens"],
                                                  n=params["num_generations"])
        response = completion.choices[0].message.content.strip()
        logger.info(f'{example_prompt} \n {self.model_name} : {response}')

        return model, tokenizer
        
    def generate(self, instance: list) -> List[str]:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=instance,
                                                  temperature=params["temperature"],
                                                  top_p=params["top_p"],
                                                  max_tokens=params["max_tokens"],
                                                  n=params["num_generations"])
        response = completion.choices[0].message.content.strip()
        return response
    
if __name__ == '__main__':
    model_api = TurboAPI()
