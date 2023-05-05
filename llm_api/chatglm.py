import os
import torch
from typing import List
from pydantic import BaseModel
from llm_api import LLMAPI, get_logger
from transformers import AutoTokenizer, AutoModel

logger = get_logger(__name__, 'INFO') # DEBUG

custom_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'ChatGLM-6B'
model_local_path = os.path.join(custom_path, model_name)
# defualt params = {"temperature": 0.95, "top_p": 0.7, "max_tokens": 2048}

class ChatGLMAPI(LLMAPI):
    def __init__(self, model_name='THUDM/chatglm-6b', model_path=model_local_path):
        super(ChatGLMAPI, self).__init__(model_name, model_path)
        self.supported_types = ['generate', 'score']
        self.name = 'chatglm'

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

        if not instance:
            return []
    
        if item.temperature == 0.0:
            item.temperature = 1e-6
            item.do_sample = False

        if type(instance) is list:
            inputs = self.tokenizer(instance, 
                                    return_tensors="pt",
                                    padding=True, 
                                    truncation=True).to("cuda")
            
            with torch.no_grad():
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
                                truncation=True,
                                max_length=2048,
                                add_special_tokens=False).to("cuda")

        prompt_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompt]
        target_lens = [len(self.tokenizer.encode(t, add_special_tokens=False)) - 1 for t in target]

        # tokenized_prompt =  self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        # tokenized_target =  self.tokenizer(target, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        # logger.warning(self.tokenizer.batch_decode(tokenized_prompt.input_ids[:,0]))
        # logger.warning(self.tokenizer.batch_decode(tokenized_prompt.input_ids[:,-1]))
        # logger.warning(tokenized_prompt.input_ids.size())
        # logger.warning(tokenized_target.input_ids.size())
        # logger.warning(prompt_lens)
        # logger.warning(target_lens)
        # logger.warning(inputs.input_ids.size())
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
            log_probs = torch.gather(input=log_probs, dim=-1, index=inputs.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        log_prob_list = []
        for i, (p_len, t_len) in enumerate(zip(prompt_lens, target_lens)):
            log_prob_list.append(log_probs[i, -t_len:].detach().cpu().tolist())
            # logger.warning(self.tokenizer.decode(inputs.input_ids[:, 1:][i, -t_len:]))

        return log_prob_list
    
if __name__ == '__main__':
    model_api = ChatGLMAPI()
