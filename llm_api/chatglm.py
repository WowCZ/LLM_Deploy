from transformers import AutoTokenizer, AutoModel
import os
import torch
from llm_api import LLMAPI
import logging
from typing import Union, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

custom_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'ChatGLM-6B'
model_local_path = os.path.join(custom_path, model_name)

params = {"temperature": 0.95, "top_p": 0.7, "max_tokens": 2048}

class ChatGLMAPI(LLMAPI):
    def __init__(self, model_name='THUDM/chatglm-6b', model_path=model_local_path):
        super(ChatGLMAPI, self).__init__(model_name, model_path)

    def _download_llm(self, model_name: str, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
            return True
        except:
            logger.error(f'failed to download model {self.model_name} into {self.model_path}')
            return False
    
    def _initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()

        # Test Case
        example_prompt = "中国有多少省份？"
        response, history = model.chat(tokenizer, 
                                       example_prompt, 
                                       history=[],
                                       max_length=params['max_tokens'],
                                       temperature=params['temperature'],
                                       top_p=params['top_p'])
        
        logger.info(f'{example_prompt} \n {self.model_name} : {response}')

        return model, tokenizer
        
    def generate(self, instance: Union[str, list]) -> List[str]:
        history = []
        if type(instance) is list:
            for i in instance:
                response, history = self.model.chat(self.tokenizer, 
                                                    i, 
                                                    history=history,
                                                    max_length=params['max_tokens'],
                                                    temperature=params['temperature'],
                                                    top_p=params['top_p'])
        else:
            response, history = self.model.chat(self.tokenizer, 
                                                instance, 
                                                history=history,
                                                max_length=params['max_tokens'],
                                                temperature=params['temperature'],
                                                top_p=params['top_p'])

        return response
    
if __name__ == '__main__':
    model_api = ChatGLMAPI()
