import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.stats
from numpy import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from copywriting import get_logger
from copywriting import human_evaluation_reader, trueskill_hotmap_reader, trueskill_gaussian_reader

logger = get_logger(__name__, 'INFO')

sns.set_theme(style="whitegrid")

api_name_map = {
    'gpt4': 'gpt-4',
    'alpaca': 'Aplaca-LoRA-7B',
    'belle': 'BELLE-7B',
    'vicuna': 'Vicuna-7B',
    'turbo': 'gpt-3.5-turbo',
    'chatglm': 'ChatGLM-6B',
    'bloom': 'BLOOM-7B1',
    'chinese-vicuna': 'Chinese-Vicuna-7B',
    'davinci': 'text-davinci-003',
    'llama': 'LLaMA-7B',
    'chinese-alpaca': 'Chinese-Alpaca-LoRA-7B',
    'moss': 'MOSS-moon-003-sft-16B',
    'vicuna-13b': 'Vicuna-13B'
}

import colorsys
import random

random.seed = 42
 
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
 
    return hls_colors
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
 
    return rgb_colors

def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

# colors = [plt.cm.Paired(i) for i in range(len(api_name_map))]
colors = list(map(lambda x: color(tuple(x)), ncolors(len(api_name_map))))
# random.shuffle(colors)
api_color_map = dict(zip(list(api_name_map.values()), colors))

llm_type_map = {
    'LLaMA': ['Aplaca-LoRA-7B', 'Chinese-Alpaca-LoRA-7B', 'Chinese-Vicuna-7B', 'LLaMA-7B', 'Vicuna-7B', 'Vicuna-13B'],
    'Bloom': ['BELLE-7B', 'BLOOM-7B1'],
    'ChatGLM': ['ChatGLM-6B'],
    'CodeGen': ['MOSS-moon-003-sft-16B'],
    'GPT': ['text-davinci-003', 'gpt-3.5-turbo', 'gpt-4'],
    'Select': ['MOSS-moon-003-sft-16B', 'Vicuna-13B', 'BELLE-7B', 'ChatGLM-6B', 'gpt-4'],
    'RankRep': ['gpt-4', 'ChatGLM-6B', 'Vicuna-13B', 'Aplaca-LoRA-7B']
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

def plot_humaneval_bar(annotated_file: str, save_fig_path: str, dump_result_path: str):
    sns.palplot(sns.color_palette("hls", 12))

    assert os.path.exists(annotated_file), f'{annotated_file} is not found!'

    for _, ds, _ in os.walk(annotated_file):
        for d in ds:
            _, analysis_results = human_evaluation_reader(os.path.join(annotated_file, d))

            bar_df = {
                'metric': [],
                'score_mean': [],
                'score_std': [],
                'api_name': [],
                'zh_metric': []
            }
            api_score_map = {}
            for t_name, a_data in analysis_results.items():
                for _, m_data in a_data.items():
                    for a, s in zip(m_data['api_name'], m_data['score_mean']):
                        if a not in api_score_map:
                            api_score_map[a] = 0
                        api_score_map[a] += s
                    bar_df['metric'].extend(m_data['metric'])
                    bar_df['zh_metric'].extend(m_data['zh_metric'])
                    bar_df['score_mean'].extend(m_data['score_mean'])
                    bar_df['score_std'].extend(m_data['score_std'])
                    bar_df['api_name'].extend(m_data['api_name'])

            metrics = list(set(bar_df['zh_metric']))
            google_sheet = dict()
            for zh_m in metrics:
                google_sheet[zh_m] = {}
            for api, score, zh_m in zip(bar_df['api_name'], bar_df['score_mean'], bar_df['zh_metric']):
                if api not in google_sheet[zh_m]:
                    google_sheet[zh_m][api] = score

            google_sheet = pd.DataFrame(google_sheet)
            google_sheet.to_csv(f'{dump_result_path}/{t_name}-google-sheet.csv')

            # hue_order = list(set(bar_df['api_name']))
            # hue_order.sort()
            hue_order = sorted(api_score_map.items(), key=lambda x:x[1], reverse=True)
            hue_order = [k for k, _ in dict(hue_order).items()]

            bar_df = pd.DataFrame(bar_df)
            bar_df.sort_values(by='metric', inplace=True, ascending=True)

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
            plt.savefig(f"{save_fig_path}/{t_name}-bar.png",dpi=600)


def plot_humaneval_radar(annotated_file: str, save_fig_path: str):
    assert os.path.exists(annotated_file), f'{annotated_file} is not found!'

    colors = ['b', 'g']

    for _, ds, _ in os.walk(annotated_file):
        for d in ds:
            _, analysis_results = human_evaluation_reader(os.path.join(annotated_file, d))


            for attr, api in analysis_results.items():
                api_name_score = {}
                for api_name, metrics in api.items():
                    metric_name = metrics['zh_metric']
                    api_name_score[api_name] = metrics['score_mean']

                api_names = sorted(api_name_score.items(), key=lambda x:x[1], reverse=True)
                api_names = [k for k, _ in dict(api_names).items()]
                if len(api_names) < 2:
                    continue

                metric_name = [f'{i}' for i, _ in enumerate(metric_name)]
                labels = np.array(metric_name)
                labels = np.concatenate((labels,[labels[0]]))
                nAttr = len(metric_name)

                angles = np.linspace(0, 2*np.pi, nAttr, endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                
                plt.figure(facecolor="white")
                ax = plt.subplot(111, polar=True)
                ax.set_ylim(0,5)

                api_color_map = {}
                for i, api_name in enumerate(api_names):
                    api_color_map[api_name] = colors[int(i%len(colors))]

                for i, api_name in enumerate(api_names):
                    if i == 0:
                        plt.thetagrids(angles*180/np.pi, labels)

                    api_score = api_name_score[api_name]
                    data = np.array(api_score)
                    data = np.concatenate((data, [data[0]]))

                    ## show api name
                    # plt.plot(angles,data,'-',color=api_color_map[api_name],linewidth=1,alpha=0.5,label=api_name)
                    # plt.fill(angles,data,facecolor=api_color_map[api_name],alpha=0.05)
                    plt.plot(angles, data, '-', linewidth=1, alpha=0.5, label=api_name)
                    plt.fill(angles, data, alpha=0.05)

                ## show title
                # plt.figtext(0.515, 0.95, attr, ha='center')
                plt.figtext(0.515, 0.95, 'Human Evaluation', ha='center')

                plt.legend(loc='lower right', prop={'size': 6})
                plt.grid(True)
                plt.savefig(f"{save_fig_path}/{attr}-radar.png" ,dpi=600)


def plot_ability_radar(annotated_file: str, save_fig_path: str, radar_name: str):
    selected_api_names = llm_type_map[radar_name]

    assert os.path.exists(annotated_file), f'{annotated_file} is not found!'

    ability_api_score = dict()
    for _, ds, _ in os.walk(annotated_file):
        for d in ds:
            _, analysis_results = human_evaluation_reader(os.path.join(annotated_file, d))

            for abli, api in analysis_results.items():
                ability_api_score[abli] = {}
                for api_name, metrics in api.items():
                    api_name = api_name_map[api_name]
                    if api_name not in selected_api_names:
                        continue
                    if api_name not in ability_api_score[abli]:
                        ability_api_score[abli][api_name] = []

                    ability_api_score[abli][api_name].extend(metrics['score_mean'])

    abli_names = [n for n in ability_api_score.keys()]
    api_ability_score = dict()
    for abli in abli_names:
        api_score = ability_api_score[abli]
        for api, score in api_score.items():
            if api not in api_ability_score:
                api_ability_score[api] = []
            
            api_ability_score[api].append(mean(score))

    print(api_ability_score)
    
    print(abli_names)
    abli_names = [f'{i}' for i, _ in enumerate(abli_names)]
    print(abli_names)

    labels = np.array(abli_names)
    labels = np.concatenate((labels,[labels[0]]))
    nAttr = len(abli_names)

    angles = np.linspace(0, 2*np.pi, nAttr, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    plt.figure(facecolor="white")
    ax = plt.subplot(111, polar=True)
    ax.set_ylim(0,5)

    api_names = [n for n in api_ability_score.keys()]
    api_score_map = dict()
    for i, api_name in enumerate(api_names):
        api_score = api_ability_score[api_name]
        api_score_map[api_name] = mean(api_score)

    api_names = sorted(api_score_map.items(), key=lambda x:x[1], reverse=True)
    api_names = [k for k, _ in dict(api_names).items()]

    api_score_map = dict()
    for i, api_name in enumerate(api_names):
        if i == 0:
            plt.thetagrids(angles*180/np.pi, labels)

        api_score = api_ability_score[api_name]
        data = np.array(api_score)
        data = np.concatenate((data, [data[0]]))

        ## show api name
        plt.plot(angles, data, '-', color=api_color_map[api_name], linewidth=1, alpha=0.5, label=api_name)
        plt.fill(angles, data, facecolor=api_color_map[api_name], alpha=0.02)
        # plt.plot(angles, data, '-', linewidth=1, alpha=0.5, label=api_name)
        # plt.fill(angles, data, alpha=0.00)

    ## show title
    # plt.figtext(0.515, 0.95, attr, ha='center')
    plt.figtext(0.515, 0.95, 'Human Evaluation', ha='center')

    plt.legend(loc='upper right', prop={'size': 6}, bbox_to_anchor=(1.3, 0.1))
    plt.grid(True)
    plt.savefig(f"{save_fig_path}/{radar_name}-radar.png" ,dpi=600)

def plot_hotmap(file_path: str, save_fig_path: str, save_fig_name: str = None):
    for _, ds, _ in os.walk(file_path):
        if len(ds) == 0:
            continue

        for d in tqdm(ds):
            # plt.clf()
            sns.set(font_scale=1.5)
            hotmap_path = os.path.join(file_path, d)
            df = trueskill_hotmap_reader(hotmap_path)
            if df is None:
                continue

            sns.set_context({"figure.figsize":(20,20)})
            ax = sns.heatmap(data=df, cmap="RdBu_r", center=50, fmt=".2f", annot=True, linewidths=0.6) 

            ax.xaxis.tick_top()
            plt.xticks(rotation=50, fontsize=15)
            plt.yticks(rotation=50, fontsize=15)

            plt.xlabel('Defender', fontsize=25, fontweight='bold')
            plt.ylabel('Offender', fontsize=25, fontweight='bold')

            if save_fig_name:
                plt.savefig(f"{save_fig_path}/{d}-{save_fig_name}.png", dpi=600)
            else:
                plt.savefig(f"{save_fig_path}/{d}-hotmap.png", dpi=600)
            
            plt.clf()

def plot_gaussian(file_path: str, save_fig_path: str, save_fig_name: str = None):

    def normal_dis(x, mu=50, sigma=5):
        return 0.3989422804014327 / sigma * math.exp(- (x - mu) * (x - mu) / (2 * sigma * sigma))

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    selected_api = llm_type_map['RankRep']
    for _, ds, _ in os.walk(file_path):
        if not ds:
            continue

        for d in tqdm(ds):
            gaussian_path = os.path.join(file_path, d)
            gaussian_map = trueskill_gaussian_reader(gaussian_path)
            if gaussian_map is None:
                continue

            for it, it_apis in gaussian_map.items():
                plt.clf()
                for api, mu_sigma in it_apis.items():
                    if api not in selected_api:
                        continue
                    # print(mu_sigma)
                    mu, sigma = mu_sigma

                    xs = [x for x in range(101)]
                    ys = [normal_dis(x, mu, sigma) for x in xs]
                    # plt.plot(xs, ys, color='black', linewidth=1.0)

                    # 填充函数区域
                    xs = np.linspace(mu - sigma*3, mu + sigma*3, 100)
                    ys = scipy.stats.norm.pdf(xs, mu, sigma)
                    plt.fill_between(xs, ys, 0, alpha=0.3, color=api_color_map[api])

                    # 绘制最上方的单个的 \mu    
                    plt.text(mu, normal_dis(mu, mu, sigma) + 0.0003, api, fontsize=4, color='black')  

                    # plt.title(f'Normal Distribution($\mu={round(mu, 2)}, \sigma={round(sigma, 2)}$)',fontsize=16)
                    plt.title(f'Normal Distribution: {d.upper()}',fontsize=16)
                    # #设置图表标题和标题字号
                    # plt.tick_params(axis='both',which='major',labelsize=14)

                    plt.xlabel('$\mu$',fontsize=10)
                    plt.ylabel('Prob.',fontsize=10)
                    plt.ylim(0, 0.6)
                    plt.xlim(0, 60)
                    # plt.legend()
                    plt.grid(color='black', alpha=0.2)
                    
                save_path = os.path.join(save_fig_path, d)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                if save_fig_name:
                    plt.savefig(f"{save_path}/{d}-{it}-{save_fig_name}.png" ,dpi=600)
                else:
                    plt.savefig(f"{save_path}/{d}-{it}-gaussian.png" ,dpi=600)

def plot_dynamic_gif(file_path: str, save_fig_path: str, save_fig_name: str = None):
    import imageio.v2 as imageio

    for _, ds, _ in  os.walk(file_path):
        if not ds:
            continue

        for d in tqdm(ds):
            img_file_path = os.path.join(file_path, d)
            for _, _, img_paths in os.walk(img_file_path):
                gif_images = []
                img_paths = [n for n in img_paths if '.png' in n]
                values = []
                for n in img_paths:
                    v = int(n.split('-')[2])
                    values.append(v)

                img_paths = dict(zip(img_paths, values))

                img_paths = sorted(img_paths.items(), key=lambda x:x[1])
                img_paths = [k for k, _ in dict(img_paths).items()]

                save_path = os.path.join(save_fig_path, d)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                logger.info(img_paths)
                for path in img_paths:
                    gif_images.append(imageio.imread(os.path.join(img_file_path, path)))
                
                if save_fig_name:
                    imageio.mimsave(f"{save_path}/{d}-{save_fig_name}.gif", gif_images, 'GIF', duration=500, floop=sys.maxsize)
                else:
                    imageio.mimsave(f"{save_path}/{d}.gif", gif_images, 'GIF', duration=500, floop=sys.maxsize)
