import os
import openai
from llm_api import LLMAPI
import logging
from typing import Union, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

openai_key = os.environ['OPENAI_KEY1'] if 'OPENAI_KEY1' in os.environ else None
assert openai_key, "No OpenAI API key detected in the environment"

params = {"temperature": 1.0, "top_p": 1.0, "num_generations": 1, "max_tokens": 512}

class DavinciAPI(LLMAPI):
    def __init__(self, model_name='text-davinci-003', model_path=None):
        super(DavinciAPI, self).__init__(model_name, model_path)
    
    def _initialize_llm(self):
        tokenizer = None
        model = None

        # Testcase
        example_prompt = "中国有多少省份？"
        openai.api_key = openai_key
        completion = openai.Completion.create(model="text-davinci-003",
                                                prompt=example_prompt,
                                                temperature=params["temperature"],
                                                top_p=params["top_p"],
                                                max_tokens=params["max_tokens"],
                                                n=params["num_generations"])
        response = completion.choices[0].text.strip()
        logger.info(f'{example_prompt} \n {self.model_name} : {response}')

        return model, tokenizer
        
    def generate(self, instance: Union[str, list]) -> List[str]:
        openai.api_key = openai_key
        completion = openai.Completion.create(model="text-davinci-003",
                                                prompt=instance,
                                                temperature=params["temperature"],
                                                top_p=params["top_p"],
                                                max_tokens=params["max_tokens"],
                                                n=params["num_generations"])
        davinci_reply = completion.choices[0].text.strip()
        return davinci_reply
    
if __name__ == '__main__':
    model_api = DavinciAPI()
