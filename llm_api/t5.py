from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from llm_api import LLMAPI
import logging
from typing import Union, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

pretrained_name = 'google/flan-t5-xxl'
model_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'FLAN-T5-11B'

model_local_path = os.path.join(model_path, model_name)


class T5API(LLMAPI):
    def __init__(self, model_name='google/flan-t5-xxl', model_path=model_local_path):
        super(T5API, self).__init__(model_name, model_path)

    def _download_llm(self, model_name: str, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
            return True
        except:
            logger.error(f'failed to download model {self.model_name} into {self.model_path}')
            return False
    
    def _initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to("cuda")

        return model, tokenizer
        
    def generate(self, instance: Union[str, list]) -> List[str]:
        if type(instance) is list:
            inputs = self.tokenizer(' '.join(instance), return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            inputs = self.tokenizer(' '.join(instance), return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return response
    
if __name__ == '__main__':
    model_api = T5API()

    # Test Case
    example_prompt = "中国有多少省份？"
    response = model_api.generate(example_prompt)
    logger.info(f'{example_prompt} \n {model_api.model_name} : {response}')
