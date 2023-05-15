import os
import random
import argparse
import pandas as pd
from copywriting import get_logger
from copywriting import sample_instance, sample_chinese_testing

random.seed = 42

logger = get_logger(__name__, 'INFO')

parser = argparse.ArgumentParser(description='Sample Generated Data!')
parser.add_argument('--step', type=int, default=1, help='Step of sampling generation data')
parser.add_argument('--file_path', type=str, default='copywriting/data', help='Path of inference data')
parser.add_argument('--dump_save_path', type=str, default='copywriting/annotated/sample_data', help='Saved path of the sampled data')
parser.add_argument('--dump_recovery_path', type=str, default='copywriting/annotated/sample_recovery_data', help='Saved path of the recovery data')
parser.add_argument('--test_case', type=int, default=24, help='Testing case number from Chinese ability testing.')
parser.add_argument('--sample_num', type=int, default=4, help='Sampled number from each LLM generated data.')
parser.add_argument('--sample_llm_num', type=int, default=4, help='Sampled LLM number.')
parser.add_argument('--llm_names', nargs='+', type=str, default=['alpaca', 'belle', 'bloom', 'chatglm', 'chinese-alpaca', 'chinese-vicuna', 'davinci', 'llama', 'vicuna', 'moss', 'turbo', 'vicuna-13b', 'gpt4'], help='LLM names')
parser.add_argument('--tasks', nargs='+', type=str, default=['empathy', 'hinting', 'humor', 'philosophical', 'poetry', 'reading', 'reasoning'], help='Human evaluation tasks')
args = parser.parse_args()
# 'empathy', 'hinting', 'humor', 'philosophical', 'poetry', 'reading', 'reasoning', 'safety', 'story', 'writing'

step = args.step
file_path = args.file_path
dump_save_path = args.dump_save_path
dump_recovery_path = args.dump_recovery_path
test_case = args.test_case
sample_num = args.sample_num
sample_llm_num = args.sample_llm_num
llm_names = args.llm_names
tasks = args.tasks

if step == 1:
    sample_stistics = sample_chinese_testing(test_case, 
                                            sample_num, 
                                            sample_llm_num, 
                                            file_path, 
                                            tasks, 
                                            llm_names, 
                                            dump_save_path, 
                                            dump_recovery_path)

    logger.info(sample_stistics)

if step == 2:
    for t in tasks:
        sample_instance(sample_num, file_path, t, llm_names, dump_save_path, dump_recovery_path)