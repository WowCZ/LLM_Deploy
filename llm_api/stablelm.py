import os
import torch
from typing import List
from pydantic import BaseModel
from llm_api import LLMAPI, get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

logger = get_logger(__name__, 'INFO') # DEBUG


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

ALPHA_PROMPT = system_prompt + " <|USER|>{instruction}<|ASSISTANT|>"

pretrained_name = 'stabilityai/stablelm-tuned-alpha-7b'
model_path = '/mnt/lustre/chenzhi/workspace/LLM/models'
model_name = 'Stablelm-7B'

model_local_path = os.path.join(model_path, model_name)

gen_config = dict(
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True
)

class StablelmAPI(LLMAPI):
    def __init__(self, model_name='stabilityai/stablelm-tuned-alpha-7b', model_path=model_local_path):
        super(StablelmAPI, self).__init__(model_name, model_path)
        self.supported_types = ['generate', 'score']
        self.name = 'alpha'

    def _download_llm(self, model_name: str, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
    
    def _initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(self.model_path).to("cuda")
        tokenizer.pad_token='[PAD]'
        model.half().eval()

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

        instance = [ALPHA_PROMPT.format(instruction=i) for i in instance]
        
        inputs = self.tokenizer(instance, 
                                return_tensors="pt",
                                padding=True, 
                                max_length=4096,
                                truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, 
                                          **gen_config,
                                          stopping_criteria=StoppingCriteriaList([StopOnTokens()]))
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

        prompt_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompt]
        target_lens = [len(self.tokenizer.encode(t, add_special_tokens=False)) for t in target]

        tokenized_prompt =  self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        tokenized_target =  self.tokenizer(target, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
        logger.debug(self.tokenizer.batch_decode(tokenized_prompt.input_ids[:,0]))
        logger.debug(self.tokenizer.batch_decode(tokenized_prompt.input_ids[:,-1]))
        logger.debug(tokenized_prompt.input_ids.size())
        logger.debug(tokenized_target.input_ids.size())
        logger.debug(prompt_lens)
        logger.debug(target_lens)
        logger.debug(inputs.input_ids.size())
        
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1, :]
            log_probs = torch.gather(input=log_probs, dim=-1, index=inputs.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        log_prob_list = []
        for i, (p_len, t_len) in enumerate(zip(prompt_lens, target_lens)):
            log_prob_list.append(log_probs[i, -t_len:].detach().cpu().tolist())
            logger.debug(self.tokenizer.decode(inputs.input_ids[:, 1:][i, -t_len:]))

        return log_prob_list
    
if __name__ == '__main__':
    model_api = StablelmAPI()

    # Test Case
    example_prompt = "中国有多少省份？"
    response = model_api.generate(example_prompt)
    logger.info(f'{example_prompt} \n {model_api.model_name} : {response}')

