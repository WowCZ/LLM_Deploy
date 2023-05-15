import random
import argparse
import pandas as pd
from copywriting import get_logger
from copywriting import recovery_score, recovery_chinese_test
from plots import plot_bar, plot_scatter

random.seed = 42

logger = get_logger(__name__, 'INFO')

parser = argparse.ArgumentParser(description='Recover Annotated Data!')
parser.add_argument('--step', type=int, default=1, help='Step of sampling generation data')
parser.add_argument('--file_path', type=str, default='copywriting/annotated_data/chinese_test_annotation', help='Path of inference data')
parser.add_argument('--dump_save_path', type=str, default='copywriting/analysis_data', help='Saved path of the sampled data')
parser.add_argument('--recovery_path', type=str, default='copywriting/sample_recovery_data', help='Saved path of the recovery data')
parser.add_argument('--recovery_task', nargs='+', type=str, default=['empathy'], help='Human evaluation tasks')
args = parser.parse_args()

step = args.step
file_path = args.file_path
dump_save_path = args.dump_save_path
recovery_path = args.recovery_path
recovery_task = args.recovery_task

if step == 1:
    recover_chinese = recovery_chinese_test(file_path, recovery_path, dump_save_path)
    plot_scatter(recover_chinese, save_fig_path='plots/Figures', save_name='chinese_capability')
    print(recover_chinese)

if step == 2:
    for t in recovery_task:
        recovery_data = recovery_score(t, file_path, recovery_path, dump_save_path)
        plot_bar(recovery_data, save_fig_path='plots/Figures', save_name='lab_triple_gpt4')
        print(recovery_data)