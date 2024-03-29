import argparse
from cli import recovery, sample, plot

def default_sampling(args):
    sample(args.name, 
           args.original_file_path, 
           args.annotating_path, 
           args.dump_recovery_path, 
           args.single_sample_size, 
           args.sample_num, 
           args.sample_llm_num, 
           args.llm_candidations, 
           args.evaluation_tasks,
           args.match_plan,
           args.seed)

def default_recovery(args):
    recovery(args.name, 
             args.annotated_data_path, 
             args.dump_result_path, 
             args.recovery_info_path, 
             args.recovery_tasks,
             args.seed)
    
def default_plot(args):
    plot(args.type,
         args.data_file,
         args.save_fig_path,
         args.save_fig_name)

parser = argparse.ArgumentParser(description='Operations on annotated data!')
subparsers = parser.add_subparsers(help='Sampling or Recovery or Plot')

sampling_parser = subparsers.add_parser(name='sampling', description='Sample inference data!')
sampling_parser.add_argument('--name', type=str, default='trueskill_evaluation', help='Name of sampling generation data')
sampling_parser.add_argument('--original_file_path', type=str, default='resource/data', help='Path of inference data')
sampling_parser.add_argument('--annotating_path', type=str, default='resource/annotated/sample_data', help='Saved path of the sampled data')
sampling_parser.add_argument('--dump_recovery_path', type=str, default='resource/annotated/sample_recovery_data', help='Saved path of the recovery data')
sampling_parser.add_argument('--single_sample_size', type=int, default=24, help='Testing case number from Chinese ability testing.')
sampling_parser.add_argument('--sample_num', type=int, default=4, help='Sampled number from each LLM generated data.')
sampling_parser.add_argument('--sample_llm_num', type=int, default=4, help='Sampled LLM number.')
sampling_parser.add_argument('--llm_candidations', nargs='+', type=str, default=['alpaca', 'belle', 'bloom', 'chatglm', 'chinese-alpaca', 'chinese-vicuna', 'davinci', 'llama', 'vicuna', 'moss', 'turbo', 'vicuna-13b', 'gpt4'], help='LLM names')
sampling_parser.add_argument('--evaluation_tasks', nargs='+', type=str, default=['empathy', 'hinting', 'humor', 'philosophical', 'poetry', 'reading', 'reasoning', 'safety', 'story', 'writing'], help='Human evaluation tasks')
sampling_parser.add_argument('--match_plan', nargs='+', type=str, default=['alpaca&belle', 'alpaca&bloom'], help='Human evaluation tasks')
sampling_parser.add_argument('--seed', type=int, default=42, help='random seed')
sampling_parser.set_defaults(func=default_sampling)

recovery_parser = subparsers.add_parser(name='recovery', description='Recover annotated data!')
recovery_parser.add_argument('--name', type=str, default='chinese_capability', help='Name of sampling generation data')
recovery_parser.add_argument('--annotated_data_path', type=str, default='resource/annotated_data/chinese_capability', help='Path of inference data')
recovery_parser.add_argument('--dump_result_path', type=str, default='resource/analysis_data', help='Saved path of the sampled data')
recovery_parser.add_argument('--recovery_info_path', type=str, default='resource/annotated_data/sample_recovery_data', help='Saved path of the recovery data')
recovery_parser.add_argument('--recovery_tasks', nargs='+', type=str, default=['empathy'], help='Human evaluation tasks')
recovery_parser.add_argument('--seed', type=int, default=42, help='random seed')
recovery_parser.set_defaults(func=default_recovery)

plot_parser = subparsers.add_parser(name='plot', description='Plot figures!')
plot_parser.add_argument('--type', type=str, default='hotmap', help='plot type')
plot_parser.add_argument('--data_file', type=str, default='path of the statistical data', help='statistical data')
plot_parser.add_argument('--save_fig_path', type=str, default='plots/figures', help='Saved path of the ploted figure')
plot_parser.add_argument('--save_fig_name', type=str, default='TrueSkill-hotmap', help='Saved figure name')
plot_parser.set_defaults(func=default_plot)
args = parser.parse_args()

args.func(args)