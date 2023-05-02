import json
import tqdm
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

def process_default_task(data_file: str) -> list:
    poetry_instruction = []
    with open(data_file, 'r') as fr:
        poetry = fr.readlines()
        for p in poetry:
            poetry_instruction.append(p.strip())
    
    poetry_instruction = [{'prompt': p} for p in poetry_instruction]

    out_data_file = data_file.replace('_instruction', '').replace('.txt', '.json')
    with open(out_data_file, 'w') as fw:
        json.dump(poetry_instruction, fw, indent=4, ensure_ascii=False)
    
    return poetry_instruction

def visit_llm_api(data_file: str, llm_url: Union[str, List[str]], llm_name: str, batch_size: int):
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
    # humor_file = '/mnt/lustre/chenzhi/workspace/LLM/copywriting/data/humor_origin.txt'
    # humor_sentences = process_humor(humor_file)
    # print(humor_sentences[0])
    # print(humor_sentences[-1])
    # print(len(humor_sentences))

    # story_file = '/mnt/lustre/chenzhi/workspace/LLM/copywriting/data/story_instruction.txt'
    # stories = process_default_task(story_file)
    # print(stories[0])
    # print(stories[-1])
    # print(len(stories))

    # poetry_file = '/mnt/lustre/chenzhi/workspace/LLM/copywriting/data/poetry_instruction.txt'
    # poetries = process_default_task(poetry_file)
    # print(poetries[0])
    # print(poetries[-1])
    # print(len(poetries))

    data_file = '/mnt/lustre/chenzhi/workspace/LLM/copywriting/data/reading.json'
    llm_name = 'moss'
    api_info = {
        'bloom': {
            'urls': [
                
            ]
        },
        'chatglm': {
            'urls': [

            ]
        },
        'davinci': {
            'urls': [
                'http://43.130.133.215:6501/generate', 
                'http://43.130.133.215:6502/generate', 
                'http://43.130.133.215:6503/generate', 
                'http://43.130.133.215:6504/generate'
            ]
        },
        'turbo': {
            'urls': [
                'http://43.130.133.215:6505/generate', 
                'http://43.130.133.215:6506/generate', 
                'http://43.130.133.215:6507/generate', 
                'http://43.130.133.215:6508/generate'
            ]
        },
        'moss': {
            'urls': [
                'http://10.140.24.70:7097/generate',
                'http://10.140.24.30:5542/generate',
                'http://10.140.24.30:8602/generate',
                'http://10.140.24.61:7817/generate',
                'http://10.140.24.61:7815/generate',
                'http://10.140.24.61:7629/generate'
            ]
        }
    }

    urls = api_info[llm_name]['urls']
    visit_llm_api(data_file, urls, llm_name, len(urls))