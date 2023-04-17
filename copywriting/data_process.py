import os, sys
import json
from typing import Union, List
from visit_api import visit_llm

header = {'Content-Type': 'application/json'}

def process_humor(data_file: str) -> list:
    with open(data_file, 'r') as fr:
        humors = fr.readlines()
        humor_sentences = []
        for hl in humors:
            hl = hl.split('.')[-1].strip()

            if len(hl) <= 3:
                continue

            if hl not in humor_sentences:
                humor_sentences.append(hl)
    
    prompt_temp = "有这样一个故事，“【故事】”，请问这个故事的笑点在哪儿？"

    humor_sentences = [{'humor_sentence': hl, 'prompt_template': prompt_temp, 'prompt': prompt_temp.replace('【故事】', hl)} for hl in humor_sentences]

    out_data_file = data_file.replace('_origin', '').replace('.txt', '.json')
    with open(out_data_file, 'w') as fw:
        json.dump(humor_sentences, fw, indent=4, ensure_ascii=False)

    return humor_sentences

def visit_llm_api(data_file: str, url_ip: Union[str, List[str]], llm_name: str, port: Union[str, List[str]]):
    with open(data_file, 'r') as fr:
        prompts = json.load(fr)

    if type(port) is str:
        batch_size = 1
    else:
        batch_size = len(port)

    batch_nums = len(prompts) // batch_size + 1 if len(prompts) % batch_size != 0 else len(prompts) // batch_size
    for i in range(batch_nums):
        if batch_size == 1:
            data = prompts[i]
        else:
            data = prompts[i*batch_size: (i+1)*batch_size]

        response = visit_llm(url_ip, header, port, data)

        if type(response) is dict:
            prompts[i][f'{llm_name}_output'] = response['output']
        else:
            assert type(response) is list
            left = i*batch_size
            right = min((i+1)*batch_size, len(prompts))
            for j in range(left, right):
                prompts[j][f'{llm_name}_output'] = response[j-left]['output']

    out_data_file = data_file.replace('.json', f'_{llm_name}.json')
    with open(out_data_file, 'w') as fw:
        json.dump(prompts, fw, indent=4, ensure_ascii=False)

    return prompts


if __name__ == '__main__':
    # humor_file = '/mnt/lustre/chenzhi/workspace/LLM/copywriting/data/humor_origin.txt'
    # humor_sentences = process_humor(humor_file)
    # print(humor_sentences[0])
    # print(humor_sentences[-1])
    # print(len(humor_sentences))

    data_file = '/mnt/lustre/chenzhi/workspace/LLM/copywriting/data/humor.json'
    url_ip = '10.140.24.72'
    port = ['6001', '6002']
    llm_name = 'bloom'
    visit_llm_api(data_file, url_ip, llm_name, port)