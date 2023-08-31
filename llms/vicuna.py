import os
import torch
from typing import List
from pydantic import BaseModel
from llms import LLMAPI, get_logger, model_download_path
from transformers import LlamaForCausalLM, LlamaTokenizer

logger = get_logger(__name__, 'INFO') # DEBUG

model_version_7b = 'Vicuna-7B'
model_version_13b = 'Vicuna-13B'

model_version_map = {
    'default': os.path.join(model_download_path, model_version_7b),
    '7b': os.path.join(model_download_path, model_version_7b),
    '13b': os.path.join(model_download_path, model_version_13b)
}

version_nickname_map = {
    'default': 'vicuna',
    '7b': 'vicuna',
    '13b': 'vicuna-13b',
}

EVAL_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"


class VicunaAPI(LLMAPI):
    def __init__(self, 
                 model_name='lmsys/vicuna-7b-delta-v1.1', 
                 model_path=model_version_map,
                 model_version='default'):
        super(VicunaAPI, self).__init__(model_name, model_path, model_version)
        self.supported_types = ['generate', 'score']
        self.name = version_nickname_map[self.model_version]

    def _download_llm(self, model_name: str, model_path: str):
        # manual operation referred on https://github.com/lm-sys/FastChat
        # download from https://zhuanlan.zhihu.com/p/620801429
        pass
    
    def _initialize_llm(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.model_path, use_fast=False, padding_side='left')
        model = LlamaForCausalLM.from_pretrained(self.model_path).to("cuda")
        tokenizer.pad_token='[PAD]'

        return model, tokenizer
        
    def generate(self, item:BaseModel) -> List[str]:
        instance = item.prompt

        if not instance:
            return []

        if type(instance) is not list:
            instance = [instance]

        if item.temperature == 0.0:
            item.temperature = 1e-6
            item.do_sample = False

        instance = [EVAL_PROMPT.format(instruction=i) for i in instance]
        
        inputs = self.tokenizer(instance, 
                                return_tensors="pt",
                                padding=True, 
                                truncation=True,
                                max_length=2048).to("cuda")
        
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
        target_lens = [t-1 for t in target_lens]

        logger.debug(self.tokenizer.batch_decode(inputs.input_ids))
        logger.debug(self.tokenizer.batch_decode(tokenized_prompt.input_ids))
        logger.debug(self.tokenizer.batch_decode(tokenized_target.input_ids[:, 1:]))
        logger.debug(prompt_lens)
        logger.debug(target_lens)
        logger.debug(inputs.input_ids.size())
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
            log_probs = torch.gather(input=log_probs, dim=-1, index=inputs.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        log_prob_list = []
        for i, (p_len, t_len) in enumerate(zip(prompt_lens, target_lens)):
            log_prob_list.append(log_probs[i, p_len-1:p_len-1+t_len].detach().cpu().tolist())
            logger.debug(self.tokenizer.decode(inputs.input_ids[:, 1:][i, p_len-1:p_len-1+t_len]))

        return log_prob_list
    
if __name__ == '__main__':
    model_api = VicunaAPI()

    # Test Case
    example_prompt = "中国有多少省份？"
    response = model_api.generate(example_prompt)
    logger.info(f'{example_prompt} \n {model_api.model_name} : {response}')
