import os
from analysis import get_logger, visit_llm_api

logger = get_logger(__name__, 'INFO')


def api_client(llm_name: str, batch_size: int, max_length: int, url_path: str, evaluation_tasks: list, inference_path: str):
    assert os.path.exists(url_path), f'Url path: {url_path} is not given.'

    server_info_file = f'{url_path}/{llm_name}_server_info.txt'
    with open(server_info_file, 'r') as fr:
        ulrs = fr.readlines()
        ulrs = [u.strip() for u in ulrs]

    gen_urls = [u for u in ulrs if 'generate' in u]

    logger.info(f'>>> Generation urls on LLM #{llm_name.upper()}# for human evaluation tasks:')
    logger.info(gen_urls)

    data_file = '{data_path}/{human_eval_task}.json'

    last_eval_tasks = []
    for _, _, fs in os.walk('copywriting/data'):
        for t in evaluation_tasks:
            if f'{t}_{llm_name}.json' in fs:
                continue

            last_eval_tasks.append(t)

    logger.info(f'>>> Unprocessed human evaluation tasks on LLM #{llm_name.upper()}#:')
    logger.info(last_eval_tasks)

    for t in last_eval_tasks:
        visit_llm_api(data_file.format(data_path=inference_path, human_eval_task=t), gen_urls, llm_name, len(gen_urls)*batch_size, max_length)
        logger.info(f'>>> LLM #{llm_name.upper()}# has been processed on task #{t}#.')