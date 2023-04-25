from transformers import AutoTokenizer, AutoModel
import os
from llm_api import LLMAPI
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

custom_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'ChatGLM-6B'
model_local_path = os.path.join(custom_path, model_name)
# defualt params = {"temperature": 0.95, "top_p": 0.7, "max_tokens": 2048}

class ChatGLMAPI(LLMAPI):
    def __init__(self, model_name='THUDM/chatglm-6b', model_path=model_local_path):
        super(ChatGLMAPI, self).__init__(model_name, model_path)

    def _download_llm(self, model_name: str, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, truncation_side='left')
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
            return True
        except:
            logger.error(f'failed to download model {self.model_name} into {self.model_path}')
            return False
    
    def _initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, truncation_side='left')
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
        return model, tokenizer
        
    def generate(self, item:BaseModel) -> List[str]:
        instance = item.prompt

        if type(instance) is list:
            # print('>>> truncation_side: ', self.tokenizer.truncation_side)
            inputs = self.tokenizer(instance, 
                                    return_tensors="pt",
                                    padding=True, 
                                    truncation=True).to("cuda")
            
            outputs = self.model.generate(**inputs, 
                                          max_new_tokens=item.max_new_tokens, 
                                          do_sample=item.do_sample, 
                                          top_p=item.top_p, 
                                          temperature=item.temperature)
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            response = [r[len(i):].strip() for i, r in zip(instance, response)]
        else:
            response, _ = self.model.chat(self.tokenizer, 
                                                instance, 
                                                history=[],
                                                max_length=item.max_new_tokens,
                                                temperature=item.temperature,
                                                top_p=item.top_p)

        return response
    
if __name__ == '__main__':
    model_api = ChatGLMAPI()
