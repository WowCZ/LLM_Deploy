import os
import json
import pandas as pd
from collections import OrderedDict
from copywriting import get_logger
from llm_api import api_name_map

logger = get_logger(__name__, 'INFO')

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
    
    prompt_temp = "有这样一个故事，“{story}”，请问这个故事的笑点在哪儿？"

    humor_sentences = [{'humor_sentence': hl, 'prompt_template': prompt_temp, 'prompt': prompt_temp.format(story=hl)} for hl in humor_sentences]

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

def _extract_human_eval(eval_item: dict) -> dict:
    user_utt = eval_item['messages'][0]['statements'][0]['utterance']
    api_reps = eval_item['messages'][1]['statements'][0]['utterance']

    evaluateData = eval_item['evaluateData']
    eval_metrics = []
    multi_annotator_scores = {}
    for i, evald in enumerate(evaluateData):
        if f'annotator_{i}' not in multi_annotator_scores:
            multi_annotator_scores[f'annotator_{i}'] = []

        for m in evald['staticIndicators']:
            multi_annotator_scores[f'annotator_{i}'].append(m['score'])

        if i == 0:
            for m in evald['staticIndicators']:
                eval_metrics.append(m['evaluateIndicator_name'])
    
    return {
        'prompt': user_utt,
        'response': api_reps,
        'multiple_scores': multi_annotator_scores,
        'eval_metrics': eval_metrics
    }


def _single_evaluation_reader(path: str) -> dict:
    panda_data = []
    name = path.split('/')[-1].split('.')[-2].split('-')
    task_name = name[0]
    api_name = '-'.join(name[1:])
    with open(path, 'r') as fr:
        line = fr.readlines()[0]
        if line[-2:].strip() == ',]':
            str_line = line[:-2].strip() + "]"
        else:
            str_line = line

    results = json.loads(str_line)

    for ditem in results:
        extracted_d = _extract_human_eval(ditem)
        multiple_scores = extracted_d['multiple_scores']
        eval_metrics = extracted_d['eval_metrics']

        for k, v in multiple_scores.items():
            for i, s in enumerate(v):
                p_data = {
                    'tName': task_name,
                    'aName': api_name,
                    'labelId': k,
                    'metricId': f'metric_{i}',
                    'scoreVal': s,
                    'metricName': eval_metrics[i]
                }
                panda_data.append(p_data)
    
    return panda_data


def human_evaluation_reader(dir_path: str) -> pd.DataFrame:
    all_panda_data = None
    for _, _, fs in os.walk(dir_path):
        for f in fs:
            if f.startswith('.'):
                continue
            panda_data = _single_evaluation_reader(os.path.join(dir_path, f))
            if all_panda_data is None:
                all_panda_data = dict()
                for k in panda_data[0].keys():
                    all_panda_data[k] = []
            
            for p in panda_data:
                for k, v in p.items():
                    all_panda_data[k].append(v)

    df = pd.DataFrame(all_panda_data)

    analysis_results = {}
    metric_map = {}
    for n in set(df['tName']):
        t_df = df[df.tName==n]
        if n not in analysis_results:
            analysis_results[n] = {}
        for a in set(t_df['aName']):
            api_df = df[(df.tName==n)&(df.aName==a)]
            if a not in analysis_results[n]:
                analysis_results[n][a] = {
                    'metric': [],
                    'score_mean': [],
                    'score_std': [],
                    'zh_metric': [],
                    'api_name': []
                }
            for m in set(api_df['metricId']):
                m_df = df[(df.tName==n)&(df.aName==a)&(df.metricId==m)]
                # print(n, a, m, m_df.scoreVal.mean())
                analysis_results[n][a]['api_name'].append(api_name_map[a])
                analysis_results[n][a]['metric'].append(m)
                if m not in metric_map:
                    metric_map[m] = list(set(m_df.metricName))[0]
                analysis_results[n][a]['zh_metric'].append(metric_map[m])
                analysis_results[n][a]['score_mean'].append(m_df.scoreVal.mean())
                analysis_results[n][a]['score_std'].append(m_df.scoreVal.std())

    logger.info(metric_map)
    return df, analysis_results


def trueskill_hotmap_reader(dir_path: str) -> pd.DataFrame:
    hotmap_file = os.path.join(dir_path, 'head_to_head_win_rate.json')
    if not os.path.exists(hotmap_file):
        logger.info(f'{hotmap_file} does not exist!')
        return None

    win_rate_hotmap = json.load(open(hotmap_file))

    off_win_rate = dict()
    win_rate_map = dict()
    for k, v in win_rate_hotmap.items():
        offe, deff = k.split('&')
        offe, deff = api_name_map[offe], api_name_map[deff] 
        if offe not in off_win_rate:
            off_win_rate[offe] = []
        off_win_rate[offe].append(v)

        if offe not in win_rate_map:
            win_rate_map[offe] = dict()
        win_rate_map[offe][deff] = float(v)

    show_order = sorted(off_win_rate.items(), key=lambda x:sum(x[1]), reverse=True)
    show_order = [k for k, _ in show_order]

    win_rate_df = {
        'Offender': [],
        'Defender': [],
        'WinRate(%)': []
    }

    for offe in show_order:
        for deff in show_order:
            if offe == deff:
                continue

            win_rate_df['Offender'].append(offe)
            win_rate_df['Defender'].append(deff)
            win_rate_df['WinRate(%)'].append(float(win_rate_map[offe][deff])*100)

    data = pd.DataFrame(win_rate_df).pivot(index="Offender", columns="Defender", values="WinRate(%)") 

    return data[show_order].reindex(show_order)


def trueskill_gaussian_reader(dir_path: str) -> dict:
    gaussian_file = os.path.join(dir_path, 'mu_sigma_by_iteration.json')
    if not os.path.exists(gaussian_file):
        logger.info(f'{gaussian_file} does not exist!')
        return None

    gaussian_statistics = json.load(open(gaussian_file))

    gaussian_map = OrderedDict()
    for itera, gaussian in enumerate(gaussian_statistics):
        gaussian_map[f'iteration-{itera}'] = dict([(api_name_map[k], v) for k, v in gaussian.items()])

    return gaussian_map