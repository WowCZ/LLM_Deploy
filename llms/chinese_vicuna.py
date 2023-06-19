import os
import torch
from typing import List
from peft import PeftModel
from pydantic import BaseModel
from llms import LLMAPI, get_logger, model_download_path
from transformers import LlamaForCausalLM, LlamaTokenizer

logger = get_logger(__name__, 'INFO') # DEBUG

pretrained_lora_name = 'Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco'
lora_model_name = 'Chinese-Vicuna-7B'
main_model_name = 'LLaMA-7B'
lora_weights = 'Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco'

model_local_path = os.path.join(model_download_path, main_model_name)
lora_local_path = os.path.join(model_download_path, lora_model_name)


CHINESE_VICUNA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

gen_config = dict(
    temperature=1e-6,
    top_k=40,
    top_p=0.9,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=1024,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0
)


class ChineseVicunaAPI(LLMAPI):
    def __init__(self, 
                 model_name='decapoda-research/llama-7b-hf', 
                 model_path=model_local_path, 
                 model_version='default',
                 adapter_name='Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco', 
                 adapter_path=lora_local_path):
        super(ChineseVicunaAPI, self).__init__(model_name, model_path, model_version, adapter_name, adapter_path)
        self.supported_types = ['generate', 'score']
        self.name = 'chinese-vicuna'

    def _download_llm(self, model_name: str, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

            tokenizer = LlamaTokenizer.from_pretrained(model_name)
            model = LlamaForCausalLM.from_pretrained(model_name)

            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            logger.info(f'>>> Download Adapter {model_name} into {model_path}.')

    def _download_adapter(self, adapter_name: str, adapter_path: str):
        if not os.path.exists(adapter_path):
            os.makedirs(adapter_path)
            model = LlamaForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16)
            tokenizer = LlamaTokenizer.from_pretrained(self.model_path)

            model_vocab_size = model.get_input_embeddings().weight.size(0)
            lora_vocab_size = len(tokenizer)

            if model_vocab_size != lora_vocab_size:
                assert model_vocab_size < lora_vocab_size
                model.resize_token_embeddings(lora_vocab_size)

            model = PeftModel.from_pretrained(
                model,
                adapter_name,
                torch_dtype=torch.float16,
            )

            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            logger.info(f'>>> Downloaded Adapter {adapter_name} into {adapter_path}.')
    
    def _initialize_llm(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.adapter_path, padding_side='left')
        tokenizer.pad_token='[PAD]'

        assert torch.cuda.is_available(), 'CPU Running Alert!'
        model = LlamaForCausalLM.from_pretrained(self.model_path,
                                                 load_in_8bit=False,
                                                 torch_dtype=torch.float16, 
                                                 low_cpu_mem_usage=True)
         
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        lora_vocab_size = len(tokenizer)

        logger.info(f'Vocab size of the base model: {model_vocab_size}')
        logger.info(f'Vocab size of the tokenizer: {lora_vocab_size}')

        if model_vocab_size != lora_vocab_size:
            assert model_vocab_size < lora_vocab_size
            logger.info(f'Resize model embeddings to fit tokenizer...')
            model.resize_token_embeddings(lora_vocab_size)

        model = PeftModel.from_pretrained(
            model,
            self.adapter_path,
            torch_dtype=torch.float16,
        ).to("cuda")

        model.eval()

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
        
        instance = [CHINESE_VICUNA_PROMPT.format(instruction=ins) for ins in instance]

        inputs = self.tokenizer(instance, 
                                return_tensors="pt",
                                padding=True, 
                                truncation=True,
                                max_length=2048).to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)

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
    model_api = ChineseVicunaAPI()

    # Test Case
    example_prompt = "中国有多少省份？"
    response = model_api.generate(example_prompt)
    logger.info(f'{example_prompt} \n {model_api.model_name} : {response}')

