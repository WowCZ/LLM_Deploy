import os
import sys
import json
import math
import numpy as np
import pandas as pd
import scipy.stats
from numpy import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from llms import api_name_map, ability_name_map, ability_en_zh_map
from . import get_logger, human_evaluation_reader, human_annotation_reader, trueskill_hotmap_reader, trueskill_gaussian_reader, trustable_humaneval_creation

ZH_FONT_PATH = os.environ['ZH_FONT_PATH']
font_path = f'{ZH_FONT_PATH}/SimHei.ttf' # ttf的路径 最好是具体路径
font_manager.fontManager.addfont(font_path)
 
# plt.rcParams['font.family'] = 'SimHei' #下面代码不行，在加上这一行
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__, 'INFO')

# logger.info(f'Supported Fonts: {font_manager.get_font_names()}')

sns.set_style("whitegrid",{"font.sans-serif":['SimHei', 'DejaVu Sans']})

import colorsys
import random

random.seed = 42

deprecated_metrics = ['符合主流价值观', '综合创意表达能力']
 
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

colors = list(map(lambda x: color(tuple(x)), ncolors(len(api_name_map))))
api_color_map = dict(zip(list(api_name_map.values()), colors))

llm_type_map = {
    'LLaMA家族': ['Aplaca-LoRA-7B', 'Chinese-Alpaca-LoRA-7B', 'Chinese-Vicuna-7B', 'LLaMA-7B', 'Vicuna-7B', 'Vicuna-13B'],
    'BLOOM家族': ['BELLE-7B', 'BLOOM-7B1'],
    'ChatGLM': ['ChatGLM-6B'],
    'CodeGen': ['MOSS-moon-003-sft-16B'],
    'GPT家族': ['text-davinci-003', 'gpt-3.5-turbo', 'gpt-4'],
    '代表模型': ['MOSS-moon-003-sft-16B', 'Vicuna-13B', 'BELLE-7B', 'ChatGLM-6B', 'gpt-4'],
    'RankRep': ['gpt-4', 'ChatGLM-6B', 'Vicuna-13B', 'Aplaca-LoRA-7B'],
    'ALL': ['ChatGLM-6B', 'Chinese-Vicuna-7B', 'LLaMA-7B', 'Vicuna-7B', 'text-davinci-003', 'Vicuna-13B', 'BELLE-7B', 'BLOOM-7B1', 'Aplaca-LoRA-7B', 'Chinese-Alpaca-LoRA-7B', 'MOSS-moon-003-sft-16B', 'gpt-3.5-turbo', 'gpt-4']
}

llm_name_gaussian = dict(zip(llm_type_map['ALL'], [0.002*i for i in range(len(llm_type_map['ALL']))]))


def plot_scatter(analysis_result_path: str, save_fig_path: str, save_name: str):
    analysis_results = json.load(open(os.path.join(analysis_result_path, '中文测评结果.json')))

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

def plot_bar(annotated_file: str, save_fig_path: str, save_fig_name: str):
    assert os.path.exists(annotated_file), f'{annotated_file} is not found!'
    logger.info(f'Deprecated Metrics: {deprecated_metrics}')

    ability_api_score = dict()
    api_score_map = {}
    for _, ds, _ in os.walk(annotated_file):
        if not ds:
            continue

        for d in tqdm(ds):
            _, analysis_results = human_evaluation_reader(os.path.join(annotated_file, d))

            for abli, api in analysis_results.items():
                ability_api_score[abli] = {}
                for api_name, metrics in api.items():
                    api_name = api_name_map[api_name]
                    if api_name not in ability_api_score[abli]:
                        ability_api_score[abli][api_name] = []

                    scores = []
                    for score, zh_metric in zip(metrics['score_mean'], metrics['zh_metric']):
                        if zh_metric in deprecated_metrics:
                            continue
                        else:
                            if api_name not in api_score_map:
                                api_score_map[api_name] = 0
                            api_score_map[api_name] += score
                            scores.append(score)

                    ability_api_score[abli][api_name].extend(scores)
    
    hue_order = sorted(api_score_map.items(), key=lambda x:x[1], reverse=True)
    hue_order = [k for k, _ in dict(hue_order).items()]

    ability_api_df = {
        '分数': [],
        '模型': [],
        '能力': []
    }
    for abli, api_scores in ability_api_score.items():
        for api, scores in api_scores.items():
            ability_api_df['能力'].append(ability_name_map[abli])
            ability_api_df['模型'].append(api)
            ability_api_df['分数'].append(mean(scores))

    ability_api_df = pd.DataFrame(ability_api_df)

    abilities_order = sorted(list(set(ability_api_df.能力)))
    abilities_order = ['创意表达能力', '言外之意理解能力', '思辨能力', '长文阅读能力', '强逻辑长文表达能力', '共情对话能力', '幽默理解能力', '安全交互能力', '古诗词鉴赏能力', '常识推理解释能力']

    logger.info(abilities_order)

    g = sns.FacetGrid(ability_api_df, 
                      hue='模型',
                      hue_order=hue_order,
                      palette=sns.color_palette([api_color_map[a] for a in hue_order]),  
                      col='能力', 
                      col_order=abilities_order,
                      col_wrap=5, 
                      height=4, 
                      ylim=(1, 5))
    g.map(sns.barplot, 
          "模型", 
          "分数", 
          order=hue_order, 
          errorbar=None)
    # g.set_xticklabels(labels = hue_order, rotation = 50, fontsize=5)
    g.set_xticklabels(labels = '')
    g.add_legend(title='模型', label_order=hue_order, ncol=1, prop={'size': 7, 'family': 'DejaVu Sans'})

    # plt.legend(ncol=2, prop={'size': 6})

    plt.savefig(f"{save_fig_path}/{save_fig_name}-bar.png",dpi=600)


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
                abli = ability_name_map[abli]
                ability_api_score[abli] = {}
                for api_name, metrics in api.items():
                    api_name = api_name_map[api_name]
                    if api_name not in selected_api_names:
                        continue
                    if api_name not in ability_api_score[abli]:
                        ability_api_score[abli][api_name] = []

                    scores = []
                    for score, zh_metric in zip(metrics['score_mean'], metrics['zh_metric']):
                        if zh_metric in deprecated_metrics:
                            continue
                        else:
                            scores.append(score)

                    ability_api_score[abli][api_name].extend(scores)

    abli_names = [n for n in ability_api_score.keys()]
    api_ability_score = dict()
    for abli in abli_names:
        api_score = ability_api_score[abli]
        for api, score in api_score.items():
            if api not in api_ability_score:
                api_ability_score[api] = []
            
            api_ability_score[api].append(mean(score))

    logger.info(f'Abilities: {abli_names}')

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

    ## show title
    if radar_name != 'ALL':
        plt.figtext(0.515, 0.95, radar_name, ha='center')

    plt.legend(loc='lower left', ncol=2, prop={'size': 4.5, 'family': 'DejaVu Sans'}, bbox_to_anchor=(-0.36, -0.1)) # bbox_to_anchor=(1.3, 0.1)
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

            sns.set_context({"figure.figsize":(25,25)})
            ax = sns.heatmap(data=df, cmap="RdBu_r", center=50, fmt=".2f", annot=True, linewidths=0.6) 

            ax.xaxis.tick_top()
            plt.xticks(rotation=50, fontsize=15)
            plt.yticks(rotation=50, fontsize=15)

            # plt.xlabel('Defender', fontsize=25, fontweight='bold')
            # plt.ylabel('Offender', fontsize=25, fontweight='bold')
            plt.xlabel('')
            plt.title('Defender')

            if save_fig_name:
                plt.savefig(f"{save_fig_path}/{d}-{save_fig_name}.png", dpi=600)
            else:
                plt.savefig(f"{save_fig_path}/{d}-hotmap.png", dpi=600)
            
            plt.clf()

def plot_gaussian(file_path: str, save_fig_path: str, save_fig_name: str = None):

    def normal_dis(x, mu=50, sigma=5):
        return 0.3989422804014327 / sigma * math.exp(- (x - mu) * (x - mu) / (2 * sigma * sigma))
    
    selected_api = llm_type_map['ALL']
    for _, ds, _ in os.walk(file_path):
        if not ds:
            continue

        for d in tqdm(ds):
            if d != 'overall':
                continue

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
                    plt.text(mu, normal_dis(mu, mu, sigma) + llm_name_gaussian[api], api, fontsize=4, color='black', fontname='DejaVu Sans')  

                    # plt.title(f'Normal Distribution($\mu={round(mu, 2)}, \sigma={round(sigma, 2)}$)',fontsize=16)
                    plt.title(f'{ability_en_zh_map[d]}(正态分布)',fontsize=16)
                    # #设置图表标题和标题字号
                    # plt.tick_params(axis='both',which='major',labelsize=14)

                    plt.xlabel('$\mu$',fontsize=10)
                    plt.ylabel('概率',fontsize=10)
                    plt.ylim(0, 0.6)
                    plt.xlim(20, 45)
                    # plt.legend()
                    plt.grid(color='black', alpha=0.2)
                    
                save_path = os.path.join(save_fig_path, d)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                if save_fig_name:
                    plt.savefig(f"{save_path}/{d}-{it}-{save_fig_name}.png", dpi=600)
                else:
                    plt.savefig(f"{save_path}/{d}-{it}-gaussian.png", dpi=600)

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


def plot_vedio(file_path: str, save_fig_path: str, save_fig_name: str = None):
    import concurrent.futures
    import imageio
    from PIL import Image
    
    def process_image(file_name):
        if file_name.endswith(".png"):
            # image = Image.open(file_name)
            image = imageio.imread(file_name)
        # return image.convert("RGB")
        return image
    
    for _, ds, _ in  os.walk(file_path):
        if not ds:
            continue
        
        for d in tqdm(ds):
            img_file_path = os.path.join(file_path, d)
            for _, _, img_paths in os.walk(img_file_path):
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
                video_images = []
                for path in img_paths:
                    video_images.append(os.path.join(img_file_path, path))

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # 利用线程池并行处理图像
                    images = list(executor.map(process_image, video_images))

                if save_fig_name:
                    video_name = f"{d}-{save_fig_name}.mp4"
                else:
                    video_name = f"{d}.mp4"

                print(type(images[0]))
                # 将图片转换为视频文件
                with imageio.get_writer(os.path.join(save_path, video_name), fps=4) as video:
                    for image in images:
                        video.append_data(image)


def plot_icc(file_path: str, save_fig_path: str, save_fig_name: str = None):
    icc_results = {
        '模型': [],
        'ICC': []
    }
    dump_trustable_instances = dict()
    for _, ds, _ in os.walk(file_path):
        if len(ds) == 0:
            continue

        for d in tqdm(ds):
            analysis_results, trustable_instances = human_annotation_reader(os.path.join(file_path, d))
            for k, v in trustable_instances.items():
                dump_trustable_instances[k] = v

            for k, vs in analysis_results.items():
                for v in vs:
                    icc_results['模型'].append(ability_name_map[k])
                    icc_results['ICC'].append(v)

    with open(os.path.join(file_path, 'trustable_instances.json'), 'w') as fw:
        json.dump(dump_trustable_instances, fw, ensure_ascii=False, indent=4)

    # with open(os.path.join(file_path, 'trustable_instances.json'), 'r') as fr:
    #     dump_trustable_instances = json.load(fr)

    # ## uncommant for recording trustable human evaluation info
    # dump_path_name = 'trustable_humaneval'
    # trustable_humaneval_creation(file_path, dump_path_name, dump_trustable_instances)

    df = pd.DataFrame(icc_results)
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(8, 6))
    # ax.set_xscale("log")

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="ICC", y="模型", data=df,
                whis=[0, 100], width=.6, palette="vlag")

    # Add in points to show each observation
    sns.stripplot(x="ICC", y="模型", data=df,
                size=2, color=".3", linewidth=0)

    # Tweak the visual presentation
    plt.yticks(fontsize=7)
    ax.xaxis.grid(True)
    plt.xlim(0.6, 1.0)
    ax.set(ylabel="")
    ax.set(xlabel="组内相关性系数（ICC）")
    sns.despine(trim=True, left=True)

    if not os.path.exists(save_fig_path):
        os.mkdir(save_fig_path)

    if save_fig_name:
        plt.savefig(f"{save_fig_path}/{save_fig_name}.png", dpi=600)
    else:
        plt.savefig(f"{save_fig_path}/icc.png", dpi=600)
            