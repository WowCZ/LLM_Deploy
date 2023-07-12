import os
import json
import copy
import tqdm
import requests
from typing import Union, List
from multiprocessing import Pool
from . import get_logger

logger = get_logger(__name__, 'INFO')

def post_data(url, header, data):
    reqs = requests.post(url=url, headers=header, data=json.dumps(data))
    while reqs.text.strip().startswith('<html><head>'):
        reqs = requests.post(url=url, headers=header, data=json.dumps(data))
    # print('#'*100)
    # print(data)
    # print('>>>>', reqs.text.strip().startswith('<html><head>'))
    # print('#'*100)
    return json.loads(reqs.text)

def _post_data(args):
    url, header, data = args
    response = post_data(url, header, data)
    if 'output' in response:
        output_key = 'output'
    else:
        assert 'outputs' in response, 'output key error from llm api!'
        output_key = 'outputs'
    pr_map = dict()
    for p, r in zip(data['prompt'], response[output_key]):
        if type(r) is list:
            r = r[0]
        pr_map[p] = r
    return pr_map

def multiprocess_post(urls, header, datas):
    pool_size = len(urls)

    assert pool_size == len(datas)
    with Pool(pool_size) as p:
        mp_reqs = p.map(_post_data, zip(urls, [header]*pool_size, datas))

    batch_pr_map = dict()
    for pr_map in mp_reqs:
        for k, v in pr_map.items():
            batch_pr_map[k] = v

    return batch_pr_map

def load_as_batches(data, api_size):
    assert len(data) >= api_size, f'The size of API {api_size} is greater than the number of data {len(data)}.'
    per_api_batch = len(data) // api_size + 1 if len(data) % api_size != 0 else len(data) // api_size
    
    batch_data = []
    prompt_temp = copy.deepcopy(data[0])
    # prompt_temp['max_new_tokens'] = 4096
    for i in range(api_size-1):
        prompt_temp['prompt'] = [p['prompt'] for p in data[i*per_api_batch:(i+1)*per_api_batch]]
        batch_data.append(copy.deepcopy(prompt_temp))
    
    prompt_temp['prompt'] = [p['prompt'] for p in data[(api_size-1)*per_api_batch:]]
    batch_data.append(copy.deepcopy(prompt_temp))
    
    return batch_data

def visit_llm(llm_url, header, data):
    # to make that the length of url is greater than the length of data
    llm_url = llm_url[: min(len(llm_url), len(data))]

    batched_data = load_as_batches(data, len(llm_url))
    batch_pr_map = multiprocess_post(llm_url, header, batched_data)
    return {'outputs': [batch_pr_map[d['prompt']] for d in data]}


def visit_llm_api(data_file: str, llm_url: Union[str, List[str]], llm_name: str, batch_size: int, max_length: int, dump_type: str, max_prompt_num: int):
    header = {'Content-Type': 'application/json'}
    
    with open(data_file, 'r') as fr:
        prompts = json.load(fr)

    if max_prompt_num:
        prompts = prompts[:max_prompt_num]

    if type(llm_url) is str:
        llm_url = [llm_url]

    out_data_file = data_file.replace('.json', f'_{llm_name}.jsonl')
    if os.path.exists(out_data_file):
        with open(out_data_file, 'r') as fr:
            out_data_len = len(fr.readlines())
    else:
        out_data_len = 0

    prompts = prompts[out_data_len:]

    p_lens = []
    for p in prompts:
        p_len = len(p['prompt'])
        p_lens.append(p_len)
        left = max(0, p_len-max_length)
        p['prompt'] = p['prompt'][left:]
    
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Number of prompts: {len(prompts)}')
    logger.info(f'Original maximum length of prompts: {max(p_lens)}')
    logger.info(f'Maximum length of prompts: {max_length}')

    if dump_type == 'incremental':
        fw = open(out_data_file, 'a')
    else:
        fw = None
    
    batch_nums = len(prompts) // batch_size + 1 if len(prompts) % batch_size != 0 else len(prompts) // batch_size
    for i in tqdm.tqdm(range(batch_nums)):
        data = prompts[i*batch_size: (i+1)*batch_size]

        response = visit_llm(llm_url, header, data)

        left = i*batch_size
        right = min((i+1)*batch_size, len(prompts))
        for j in range(left, right):
            prompts[j][f'{llm_name}_output'] = response['outputs'][j-left]
            if fw:
                print(f'>>> Add #sample-{j}')
                fw.write(json.dumps(prompts[j], ensure_ascii=False)+'\n')
                fw.flush()

    if dump_type == 'incremental':
        fw.flush()
        fw.close()
    elif dump_type == 'oncetime':
        out_data_file = out_data_file.replace('.jsonl', '.json')
        with open(out_data_file, 'w') as fw:
            json.dump(prompts, fw, indent=4, ensure_ascii=False)
    else:
        raise TypeError(f'{dump_type} for dump_type is not defined!')

    return prompts


def revisit_llm_api(data_file: str, llm_url: str, llm_name: str, output_match_condition: str):
    header = {'Content-Type': 'application/json'}
    
    revisit_data_file = data_file.replace('.json', f'_{llm_name}.json')
    with open(revisit_data_file, 'r') as fr:
        prompts = json.load(fr)

    if type(llm_url) is str:
        llm_url = [llm_url]

    revisit_cnt = 0
    output_prefix = f'{llm_name}_output'
    for p in prompts:
        if p[output_prefix] == output_match_condition:
            revisit_cnt += 1
            response = visit_llm(llm_url, header, [p])
            p[output_prefix] = response['outputs'][0]

    logger.info(f'Revisited prompts: {revisit_cnt}')

    with open(revisit_data_file, 'w') as fw:
        json.dump(prompts, fw, indent=4, ensure_ascii=False)

    return prompts