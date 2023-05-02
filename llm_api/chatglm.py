import torch
import os
from llm_api import LLMAPI
import logging
from typing import List
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

custom_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'ChatGLM-6B'
model_local_path = os.path.join(custom_path, model_name)
# defualt params = {"temperature": 0.95, "top_p": 0.7, "max_tokens": 2048}

class ChatGLMAPI(LLMAPI):
    def __init__(self, model_name='THUDM/chatglm-6b', model_path=model_local_path):
        super(ChatGLMAPI, self).__init__(model_name, model_path)
        self.supported_types = ['generate', 'score']

    def _download_llm(self, model_name: str, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, truncation_side='left')
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
    
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
    
    def score(self, item:BaseModel) -> List[List[float]]:
        prompt = item.prompt
        target = item.target

        if type(prompt) is not list:
            prompt = [prompt]
            target = [target]
        
        instance = [p+t for p, t in zip(prompt, target)]
        
        inputs = self.tokenizer(instance, 
                                return_tensors="pt",
                                padding=True,
                                padding_side='left',
                                truncation=True,
                                truncation_side='left',
                                max_length=2048).to("cuda")
        
        target_lens = [len(self.tokenizer.encode(t, add_special_tokens=False)) for t in target]
        
        with torch.no_grad():
            logits = self.model(inputs.input_ids).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.gather(input=log_probs, dim=-1, index=inputs.input_ids.unsqueeze(-1)).squeeze(-1)

        log_prob_list = []
        for i, t_len in enumerate(target_lens):
            log_prob_list.append(log_probs[i, -t_len-1: -1].detach().cpu().tolist())

        return log_prob_list
    
if __name__ == '__main__':
    model_api = ChatGLMAPI()
