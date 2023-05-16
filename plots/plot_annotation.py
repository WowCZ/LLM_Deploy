import os
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
import seaborn as sns
from copywriting import human_evaluation_reader

sns.set_theme(style="whitegrid")

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

def plot_bar(analysis_results: dict, save_fig_path: str, save_name: str, model_num: int=4):
    sns.palplot(sns.color_palette("hls", 12))

    bar_df = {
        'annotator': [],
        'score': [],
        'llm': [],
        'rank_score': []
    }

    llm_score_map = {}
    llm_rank_map = {}
    for annotator, results in analysis_results.items():
        for llm, scores in results.items():
            mean_score = mean(scores['score'])
            mean_rank_score = mean([5-(r-1)/model_num*5 for r in scores['rank']])
            bar_df['annotator'].append(annotator)
            bar_df['llm'].append(llm)
            bar_df['score'].append(mean_score)
            bar_df['rank_score'].append(mean_rank_score)

            if llm not in llm_score_map:
                llm_score_map[llm] = 0
            llm_score_map[llm] += mean_score

            if llm not in llm_rank_map:
                llm_rank_map[llm] = 0
            llm_rank_map[llm] += mean_rank_score

    hue_order = sorted(llm_score_map.items(), key=lambda x:x[1], reverse=True)
    hue_order = [k for k, _ in dict(hue_order).items()]
    # print(hue_order)
    bar_df = pd.DataFrame(bar_df)
    bar_df.sort_values(by='annotator', inplace=True, ascending=True)

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=bar_df, kind="bar",
        x="annotator", y="score", hue="llm", hue_order=hue_order,
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("", "Lab Evaluation Score")
    g.legend.set_title("LLM")

    plt.ylim(1, 5)
    plt.savefig(f"{save_fig_path}/{save_name}.png", dpi=600)


def plot_scatter(analysis_results: dict, save_fig_path: str, save_name: str):
    df_results = {
        'LLM': [],
        'GenScore': [],
        'InsScore': [],
        'type': []
    }

    llm_score_map = dict()
    for llm, scores in analysis_results.items():
        gen_scores = []
        ins_scores = []
        for s in scores:
            gen_scores.append(s['gen_score'])
            ins_scores.append(s['ins_score'])

        df_results['LLM'].append(llm)
        df_results['GenScore'].append(mean(gen_scores))
        df_results['InsScore'].append(mean(ins_scores))
        df_results['type'].append(8)
        llm_score_map[llm] = mean(gen_scores) + mean(ins_scores)

    df = pd.DataFrame(df_results)

    hue_order = sorted(llm_score_map.items(), key=lambda x:x[1], reverse=True)
    hue_order = [k for k, _ in dict(hue_order).items()]

    sns.relplot(data=df, 
                x="GenScore", 
                y="InsScore", 
                hue="LLM", 
                hue_order=hue_order, 
                style="LLM", 
                size="type",
                size_norm=(1,6), 
                kind="scatter", 
                legend=False)

    for li, llm in enumerate(df_results['LLM']):
        if df_results['type'][li] == 8:
            if llm == 'vicuna-13b':
                x_ = -0.2
                y_ = 0.2
            elif llm == 'belle':
                x_ = -0.1
                y_ = 0.2
            else:
                x_ = -0.2
                y_ = -0.2
                
            plt.annotate(api_name_map[llm], 
                            xy=(df_results['GenScore'][li], df_results['InsScore'][li]), 
                            xytext=(df_results['GenScore'][li]+x_, df_results['InsScore'][li]+y_),
                            fontsize=5)
                

    plt.vlines(x=2.5, ymin=0.5, ymax=5.2, 
           colors='red', linewidth=1)
    plt.hlines(y=2.5, xmin=0.5, xmax=5.2,
            colors='red', linewidth=1)
    
    plt.xlim(0, 5.2)
    plt.ylim(0, 5.2)
    plt.savefig(f"{save_fig_path}/{save_name}.png", dpi=600)

def plot_human_evaluation(annotated_file: str, save_fig_path: str, dump_result_path: str):
    sns.palplot(sns.color_palette("hls", 12))

    for _, ds, _ in os.walk(annotated_file):
        for d in ds:
            _, analysis_results = human_evaluation_reader(os.path.join(annotated_file, d))

            bar_df = {
                'metric': [],
                'score_mean': [],
                'score_std': [],
                'api_name': []
            }
            api_score_map = {}
            for t_name, a_data in analysis_results.items():
                for a_name, m_data in a_data.items():
                    for a, s in zip(m_data['api_name'], m_data['score_mean']):
                        if a not in api_score_map:
                            api_score_map[a] = 0
                        api_score_map[a] += s
                    bar_df['metric'].extend(m_data['metric'])
                    bar_df['score_mean'].extend(m_data['score_mean'])
                    bar_df['score_std'].extend(m_data['score_std'])
                    bar_df['api_name'].extend(m_data['api_name'])

            # hue_order = list(set(bar_df['api_name']))
            # hue_order.sort()
            hue_order = sorted(api_score_map.items(), key=lambda x:x[1], reverse=True)
            hue_order = [k for k, _ in dict(hue_order).items()]

            bar_df = pd.DataFrame(bar_df)
            bar_df.sort_values(by='metric', inplace=True, ascending=True)
            bar_df.to_csv(f'{dump_result_path}/{t_name}.csv')

            # Draw a nested barplot by species and sex
            g = sns.catplot(
                data=bar_df, kind="bar",
                x="metric", y="score_mean", hue="api_name", hue_order=hue_order,
                errorbar="sd", palette="dark", alpha=.6, height=6
            )
            g.despine(left=True)
            g.set_axis_labels("", "Human Evaluation Score")
            g.legend.set_title("LLM")

            plt.ylim(1, 5)
            plt.savefig(f"{save_fig_path}/{t_name}.png",dpi=600)