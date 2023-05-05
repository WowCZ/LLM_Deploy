import os
import argparse
from copywriting import get_logger, visit_llm_api

logger = get_logger(__name__, 'INFO')

parser = argparse.ArgumentParser(description='llm api server')
parser.add_argument('--model_name', type=str, default='alpaca', help='Supported model names: [alpaca, bloom, chatglm, davinci, llama, moss, t5, turbo, vicuna]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for each api model.')
args = parser.parse_args()

llm_name = args.model_name
batch_size = args.batch_size

server_info_file = f'copywriting/urls/{llm_name}_server_info.txt'
with open(server_info_file, 'r') as fr:
    ulrs = fr.readlines()
    ulrs = [u.strip() for u in ulrs]

gen_urls = [u for u in ulrs if 'generate' in u]

logger.info(f'>>> Generation urls on LLM #{llm_name}# for human evaluation tasks:')
logger.info(gen_urls)

data_file = 'copywriting/data/{human_eval_task}.json'
human_eval_tasks = ['bias', 'empathy', 'hinting', 'humor', 'philosophical', 'poetry', 'reading', 'story', 'writing']
extracted_tasks = ['bias', 'poetry', 'story', 'reading']

last_eval_tasks = []
for root, ds, fs in os.walk('copywriting/data'):
    for t in human_eval_tasks:
        if f'{t}_{llm_name}.json' in fs or t in extracted_tasks:
            continue

        last_eval_tasks.append(t)

logger.info(f'>>> Unprocessed human evaluation tasks on LLM #{llm_name}#:')
logger.info(last_eval_tasks)

for t in last_eval_tasks:
    visit_llm_api(data_file.format(human_eval_task=t), gen_urls, llm_name, len(gen_urls)*batch_size)
    logger.info(f'>>> LLM #{llm_name}# has been processed on task #{t}#.')