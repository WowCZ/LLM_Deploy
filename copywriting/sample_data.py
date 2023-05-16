import os
import json
import random
from typing import Any
from copywriting import get_logger
from collections import OrderedDict

random.seed = 42

logger = get_logger(__name__, 'INFO')

def sample_chinese_testing(sample_size: int, 
                           sample_case_num: int, 
                           sample_llm_num: int, 
                           file_path: str, 
                           tasks: list, 
                           llm_names: list, 
                           dump_save_path: str, 
                           dump_recovery_path: str,
                           evaluation_name: str='中文能力测评',
                           score_template: Any={
                                                    "文本生成能力": '1-5',
                                                    "任务适应性": '1-5'
                                                }):
    random.shuffle(llm_names)
    anonymous_names = [f'模型{i+1}' for i in range(len(llm_names))]
    llm_anonymous_mapping = dict(zip(llm_names, anonymous_names))
    recovery_mapping = dict(zip(anonymous_names, llm_names))
    recovery_file = os.path.join(dump_recovery_path, f'{evaluation_name}.json')
    with open(recovery_file, 'w') as fw:
        json.dump(recovery_mapping, fw, ensure_ascii=False, indent=4)

    sample_stistics = dict()
    gen_info = dict()
    for t in tasks:
        gen_info[t] = dict()
        prompt_file = os.path.join(file_path, f'{t}.json')
        gen_info[t]['prompt'] = json.load(open(prompt_file))
        for llm in llm_names:
            gen_file = os.path.join(file_path, f'{t}_{llm}.json')
            gen_info[t][llm] = json.load(open(gen_file))
    
    for c_id in range(sample_size):
        if sample_size == 1:
            test_case_name = evaluation_name
        else:
            test_case_name = f'{evaluation_name}-{c_id+1}'
        random.shuffle(tasks)
        record_case = OrderedDict()
        random.shuffle(llm_names)
        for t_id, t in enumerate(tasks[:sample_case_num]):
            case_id = random.choice(range(len(gen_info[t]['prompt'])))
            prompt = gen_info[t]['prompt'][case_id]
            record_case[f'指令任务{t_id+1}'] = prompt['prompt']
            record_case[f'模型回复{t_id+1}'] =  dict()
            for llm in llm_names[:sample_llm_num]:
                llm_gen = gen_info[t][llm][case_id][f'{llm}_output']
                record_case[f'模型回复{t_id+1}'][llm_anonymous_mapping[llm]] = llm_gen

            record_case[f'模型回复{t_id+1}'] = OrderedDict(sorted(record_case[f'模型回复{t_id+1}'].items(), key=lambda x:x[0]))

        record_case['打分'] =  OrderedDict()
        for llm_anon in record_case['模型回复1'].keys():
            record_case['打分'][llm_anon] = score_template

            if llm_anon not in sample_stistics:
                sample_stistics[llm_anon] = 0
            
            sample_stistics[llm_anon] += 1

        test_case_file = os.path.join(dump_save_path, f'{test_case_name}.json')
        with open(test_case_file, 'w') as fw:
            json.dump(record_case, fw, ensure_ascii=False, indent=4)

    sample_stistics = OrderedDict(sorted(sample_stistics.items(), key=lambda x:x[0]))

    return sample_stistics

def _sample_pairwise_game(sample_case_num: int, 
                           inference_path: str, 
                           task: str, 
                           pairwise_llm: list, 
                           dump_save_path: str, 
                           dump_recovery_path: str,
                           evaluation_name: str,
                           score_template: Any='1-5'):
    random.shuffle(pairwise_llm)
    anonymous_names = [f'模型{i+1}' for i in range(len(pairwise_llm))]
    llm_anonymous_mapping = dict(zip(pairwise_llm, anonymous_names))
    recovery_mapping = dict(zip(anonymous_names, pairwise_llm))

    if not os.path.exists(dump_recovery_path):
        os.makedirs(dump_recovery_path)

    if not os.path.exists(dump_save_path):
        os.makedirs(dump_save_path)

    recovery_file = os.path.join(dump_recovery_path, f'{evaluation_name}.json')
    with open(recovery_file, 'w') as fw:
        json.dump(recovery_mapping, fw, ensure_ascii=False, indent=4)

    gen_info = dict()
    prompt_file = os.path.join(inference_path, f'{task}.json')
    gen_info['prompt'] = json.load(open(prompt_file))
    for llm in pairwise_llm:
        gen_file = os.path.join(inference_path, f'{task}_{llm}.json')
        gen_info[llm] = json.load(open(gen_file))
    
 
    record_case = OrderedDict()
    random.shuffle(pairwise_llm)
    sampled_case_ids = []
    for q_id in range(sample_case_num):
        case_id = random.choice(range(len(gen_info['prompt'])))
        while case_id in sampled_case_ids:
            case_id = random.choice(range(len(gen_info['prompt'])))
        sampled_case_ids.append(case_id)

        prompt = gen_info['prompt'][case_id]
        record_case[f'指令任务{q_id+1}'] = prompt['prompt']
        record_case[f'模型回复{q_id+1}'] =  dict()
        for llm in pairwise_llm:
            llm_gen = gen_info[llm][case_id][f'{llm}_output']
            record_case[f'模型回复{q_id+1}'][llm_anonymous_mapping[llm]] = llm_gen

        record_case[f'模型回复{q_id+1}'] = OrderedDict(sorted(record_case[f'模型回复{q_id+1}'].items(), key=lambda x:x[0]))

    record_case['打分'] =  OrderedDict()
    for llm_anon in record_case['模型回复1'].keys():
        record_case['打分'][llm_anon] = score_template

    test_case_file = os.path.join(dump_save_path, f'{evaluation_name}.json')
    with open(test_case_file, 'w') as fw:
        json.dump(record_case, fw, ensure_ascii=False, indent=4)

    return recovery_mapping


def sample_trueskill(match_plan: list, 
                     task: str, 
                     sample_num:int, 
                     inference_path:str, 
                     dump_save_path:str, 
                     dump_recovery_path:str,
                     evaluation_name:str='TrueSkill',
                     score_template: str='1-5'):
    for gi, match in enumerate(match_plan):
        pairwise_llm = match.split('&')
        single_name = f'{evaluation_name}-game-{gi}'
        _sample_pairwise_game(sample_num, 
                           inference_path, 
                           task, 
                           pairwise_llm, 
                           dump_save_path, 
                           dump_recovery_path,
                           single_name,
                           score_template)


def recovery_trueskill(recovery_task: str, annotated_path: str, recovery_path: str, dump_result_path: str):
    result_info = {
        recovery_task: OrderedDict()
    }

    for _, _, fs in os.walk(annotated_path):
        for f in fs:
            f_name = f.split('.')[0]
            result_info[recovery_task][f_name] = dict()
            annotated_result = json.load(open(os.path.join(annotated_path, f)))["打分"]
            model_name_map = json.load(open(os.path.join(recovery_path, f)))

            winner = ''
            max_score = 0
            score = []
            for k, v in model_name_map.items():
                s = int(annotated_result[k])
                result_info[recovery_task][f_name][v] = s
                score.append(s)
                if s > max_score:
                    winner = v
                    max_score = s
            
            max_cnt = sum([1 for s in score if s==max_score])
            if max_cnt == 2:
                winner = 'draw'

            result_info[recovery_task][f_name]['winner'] = winner

    if not os.path.exists(dump_result_path):
        os.makedirs(dump_result_path)

    result_file = os.path.join(dump_result_path, f'{recovery_task}-TrueSkill-result.json')
    with open(result_file, 'w') as fw:
        json.dump(result_info, fw, ensure_ascii=False, indent=4)

    return result_info


def sample_instance(sample_num: int, 
                    file_path: str, 
                    task: str, 
                    llm_names: list, 
                    dump_save_path: str, 
                    dump_recovery_path: str):
    
    task_file = os.path.join(file_path, f'{task}.json')
    assert os.path.exists(task_file), f'{task_file} is not existed!'

    task_instances = json.load(open(task_file))
    instance_num = len(task_instances)
    sample_list = random.sample(range(0, instance_num), sample_num)

    sampled_prompts = []
    anonymous_samples = dict()
    recovery_mapping = dict()
    for s_id in sample_list:
        prompt = task_instances[s_id]['prompt']
        sampled_prompts.append(prompt)
        anonymous_samples[prompt] = []
        recovery_mapping[prompt] = dict()

    for llm_name in llm_names:
        gen_file = os.path.join(file_path, f'{task}_{llm_name}.json')
        assert os.path.exists(gen_file), f'{gen_file} is not existed!'

        gen_instances = json.load(open(gen_file))
        assert instance_num == len(gen_instances), 'The number of generated data does not equal to original data.'

        for id, s_id in enumerate(sample_list):
            prompt = sampled_prompts[id]
            gen_result = gen_instances[s_id][f'{llm_name}_output']
            anonymous_samples[prompt].append(
                {
                    'anonymous': gen_result,
                    'score': [],
                    'rank': []
                }
            )

            if gen_result not in recovery_mapping[prompt]:
                recovery_mapping[prompt][gen_result] = []

            recovery_mapping[prompt][gen_result].append(llm_name)
    
    for p in anonymous_samples.keys():
        random.shuffle(anonymous_samples[p])

    if not os.path.exists(dump_save_path):
        os.makedirs(dump_save_path)

    if not os.path.exists(dump_recovery_path):
        os.makedirs(dump_recovery_path)

    sample_file = os.path.join(dump_save_path, f'{task}_sample.json')
    recover_file = os.path.join(dump_recovery_path, f'{task}_recovery.json')

    with open(sample_file, 'w') as fw:
        json.dump(anonymous_samples, fw, ensure_ascii=False, indent=4)

    with open(recover_file, 'w') as fw:
        json.dump(recovery_mapping, fw, ensure_ascii=False, indent=4)


def recovery_chinese_test(file_path: str, recovery_path:str, dump_analysis_path:str):
    recovery_map = json.load(open(os.path.join(recovery_path, '中文能力测评.json')))
    annotation_info = {}
    for r, ds, fs in os.walk(file_path):
        for f in fs:
            if '中文能力评测' in f:
                recovery_data = json.load(open(os.path.join(file_path, f)))
                score_info = recovery_data["打分"]
                for anonymous, scores in score_info.items():
                    llm_name = recovery_map[anonymous]
                    if llm_name not in annotation_info:
                        annotation_info[llm_name] = []

                    annotation_info[llm_name].append(
                        {
                            'gen_score': int(scores['文本生成能力']),
                            'ins_score': int(scores['任务适应性'])
                        }
                    )

    if not os.path.exists(dump_analysis_path):
        os.makedirs(dump_analysis_path)

    with open(os.path.join(dump_analysis_path, '中文测评结果.json'), 'w') as fw:
        json.dump(annotation_info, fw, ensure_ascii=False, indent=4)

    return annotation_info


def recovery_score(task: str, file_path: str, recovery_path:str, dump_analysis_path:str):
    recovery_data = json.load(open(os.path.join(recovery_path, f'{task}_recovery.json')))

    task_annotation_info = {}
    for r, ds, fs in os.walk(file_path):
        for f in fs:
            if f'{task}_sample' in f:
                annotator_id = f.split('.')[0].split('_')[-1]
                task_annotation_info[annotator_id] = []
                annotated_file = os.path.join(file_path, f)
                annotated_data = json.load(open(annotated_file))
                for prompt, scores in annotated_data.items():
                    for llm_score in scores:
                        response = llm_score['anonymous']
                        llm_names = recovery_data[prompt][response]
                        score = llm_score['score'][0]
                        rank = llm_score['rank'][0]
                        for llm_name in llm_names:
                            task_annotation_info[annotator_id].append(
                                {
                                    'llm': llm_name,
                                    'score': score,
                                    'rank': rank
                                }
                            )

    final_task_annotation_info = {}
    for annotator_id, annotations in task_annotation_info.items():
        final_task_annotation_info[annotator_id] = dict()
        for annotation in annotations:
            llm = annotation['llm']
            if llm not in final_task_annotation_info[annotator_id]:
                final_task_annotation_info[annotator_id][llm] = dict()
                final_task_annotation_info[annotator_id][llm]['score'] = []
                final_task_annotation_info[annotator_id][llm]['rank'] = []
            
            final_task_annotation_info[annotator_id][llm]['score'].append(annotation['score'])
            final_task_annotation_info[annotator_id][llm]['rank'].append(annotation['rank'])


    if not os.path.exists(dump_analysis_path):
        os.makedirs(dump_analysis_path)

    with open(os.path.join(dump_analysis_path, f'{task}.json'), 'w') as fw:
        json.dump(final_task_annotation_info, fw, ensure_ascii=False, indent=4)

    return final_task_annotation_info
        
    


