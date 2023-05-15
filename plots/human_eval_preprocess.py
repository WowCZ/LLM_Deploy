import os, sys
import json
import pandas as pd
import numpy as np

api_name_map = {
    'alpaca': 'Aplaca-LoRA-7B',
    'belle': 'BELLE-7B',
    'bloom': 'BLOOM-7B',
    'chatglm': 'ChatGLM-6B',
    'chinese-alpaca': 'Chinese-Alpaca-LoRA-7B',
    'chinese-vicuna': 'Chinese-Vicuna-7B',
    'davinci': 'text-davinci-003',
    'llama': 'LLaMA-7B',
    'moss': 'MOSS-moon-003-sft-16B',
    'turbo': 'gpt-3.5-turbo',
    'vicuna': 'Vicuna-7B',
    'vicuna-13b': 'Vicuna-13B',
    'gpt4': 'gpt-4'
}

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


def _single_file_reader(path: str) -> dict:
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


def file_reader(dir_path: str) -> pd.DataFrame:
    all_panda_data = None
    for root, ds, fs in os.walk(dir_path):
        for f in fs:
            if f.startswith('.'):
                continue
            panda_data = _single_file_reader(os.path.join(dir_path, f))
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

    print(metric_map)
    return df, analysis_results


if __name__ == '__main__':
    df = file_reader('plots/HumanEvalResults')