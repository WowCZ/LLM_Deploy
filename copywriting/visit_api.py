import json
import copy
import tqdm
import requests
from typing import Union, List
from multiprocessing import Pool
from copywriting import get_logger

logger = get_logger(__name__, 'INFO')

def post_data(url, header, data):
    reqs = requests.post(url=url, headers=header, data=json.dumps(data))
    return json.loads(reqs.text)

def _post_data(args):
    url, header, data = args
    response = post_data(url, header, data)
    pr_map = dict()
    for p, r in zip(data['prompt'], response['outputs']):
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
    prompt_temp['max_new_tokens'] = 1024
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


def visit_llm_api(data_file: str, llm_url: Union[str, List[str]], llm_name: str, batch_size: int):
    header = {'Content-Type': 'application/json'}
    
    with open(data_file, 'r') as fr:
        prompts = json.load(fr)

    if type(llm_url) is str:
        llm_url = [llm_url]
    
    batch_nums = len(prompts) // batch_size + 1 if len(prompts) % batch_size != 0 else len(prompts) // batch_size
    for i in tqdm.tqdm(range(batch_nums)):
        data = prompts[i*batch_size: (i+1)*batch_size]

        response = visit_llm(llm_url, header, data)

        left = i*batch_size
        right = min((i+1)*batch_size, len(prompts))
        for j in range(left, right):
            prompts[j][f'{llm_name}_output'] = response['outputs'][j-left]

    out_data_file = data_file.replace('.json', f'_{llm_name}.json')
    with open(out_data_file, 'w') as fw:
        json.dump(prompts, fw, indent=4, ensure_ascii=False)

    return prompts


if __name__ == '__main__':
    # Single Thread
    url= 'http://43.130.133.215:6501/generate'
    header = {'Content-Type': 'application/json'}
    data = {
        "prompt": "有这样一个故事：“我：“爸在干嘛呢？最近家里生意还好吧。”爸：“已汇””，请问这个故事的笑点在哪儿？"
    }

    logger.info(post_data(url, header, data))

    # Multiprocessing
    urls = ['http://43.130.133.215:6501/generate', 'http://43.130.133.215:6502/generate']
    datas = [
        {
        "prompt": "'有这样一个故事：“我：“爸在干嘛呢？最近家里生意还好吧。”爸：“已汇””，请问这个故事的笑点在哪儿？'"
        },
        {
        "prompt": "有这样一个故事，““爸，端午节我不回家，捎几个粽子给我吧。”“行，你要哪种？”“都行，能解馋就行。”“好！”晚上回宿舍，打开邮箱发现爸爸发了一封邮件，足足4个G，下面还留言：不知道你好哪口，就每种给你发了一个。”，请问这个故事的笑点在哪儿？"
        }
    ]
    logger.info(multiprocess_post(urls, header, datas))