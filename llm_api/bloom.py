from transformers import AutoTokenizer, AutoModel, BloomForCausalLM
import os
from llm_api import LLMAPI
import logging
from typing import Union, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

pretrained_name = 'bigscience/bloomz-7b1'
model_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'BLOOM-7B1'

model_local_path = os.path.join(model_path, model_name)


class BloomAPI(LLMAPI):
    def __init__(self, model_name='bigscience/bloomz-7b1', model_path=model_local_path):
        super(BloomAPI, self).__init__(model_name, model_path)

    def _download_llm(self, model_name: str, model_path: str) -> bool:
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to("cuda")

                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
            return True
        except:
            logger.error(f'failed to download model {self.model_name} into {self.model_path}')
            return False
    
    def _initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = BloomForCausalLM.from_pretrained(self.model_path).to("cuda")

        return model, tokenizer
        
    def generate(self, instance: Union[str, list]) -> List[str]:
        if type(instance) is list:
            instance = ' '.join(instance)
            
        inputs = self.tokenizer(instance, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print(f'original output:{response}')
        response = response[len(instance):]

        return response
    
if __name__ == '__main__':
    model_api = BloomAPI()

    # Test Case
    example_prompt = "中国有多少省份？"
    response = model_api.generate(example_prompt)
    logger.info(f'{example_prompt} \n {model_api.model_name} : {response}')

