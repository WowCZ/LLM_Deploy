import os
import torch
from typing import List
from pydantic import BaseModel
from llm_api import LLMAPI, get_logger
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = get_logger(__name__, 'INFO') # DEBUG

pretrained_name = 'fnlp/moss-moon-003-sft'
model_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'MOSS-16B'

model_local_path = os.path.join(model_path, model_name)


class MOSSAPI(LLMAPI):
    def __init__(self, model_name='fnlp/moss-moon-003-sft', model_path=model_local_path):
        super(MOSSAPI, self).__init__(model_name, model_path)
        self.supported_types = ['generate', 'score']
        self.name = 'moss'

    def _download_llm(self, model_name: str, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
    
    def _initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).to("cuda")
        tokenizer.pad_token = '[PAD]'

        return model, tokenizer
        
    def generate(self, item:BaseModel) -> List[str]:
        instance = item.prompt

        if not instance:
            return []
        
        if item.temperature == 0.0:
            item.temperature = 1e-6
            item.do_sample = False
        
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

        tokenized_prompt =  self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        tokenized_target =  self.tokenizer(target, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)

        prompt_lens = tokenized_prompt.attention_mask.sum(-1).int().tolist()
        target_lens = tokenized_target.attention_mask.sum(-1).int().tolist()
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
            log_probs = torch.gather(input=log_probs, dim=-1, index=inputs.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        log_prob_list = []
        for i, (p_len, t_len) in enumerate(zip(prompt_lens, target_lens)):
            log_prob_list.append(log_probs[i, p_len-1:p_len-1+t_len].detach().cpu().tolist())

        return log_prob_list
    
if __name__ == '__main__':
    model_api = MOSSAPI()

    # Test Case
    example_prompt = "中国有多少省份？"
    response = model_api.generate(example_prompt)
    logger.info(f'{example_prompt} \n {model_api.model_name} : {response}')
