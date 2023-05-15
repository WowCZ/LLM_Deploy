import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from human_eval_preprocess import file_reader
# matplotlib.rcParams['font.family']='SimHei'
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']

df, analysis_results = file_reader('plots/HumanEvalResults/EvalPlat_subjective/安全能力')

colors = ['b', 'g']

for attr, api in analysis_results.items():
    api_name_score = {}
    api_names = []
    for api_name, metrics in api.items():
        metric_name = metrics['zh_metric']
        api_names.append(api_name)
        api_name_score[api_name] = metrics['score_mean']

    api_names.sort()
    if len(api_names) < 2:
        continue

    print('#'*50)
    print(metric_name)
    # uncommant
    metric_name = [f'{i}' for i, _ in enumerate(metric_name)]
    print(metric_name)
    
    labels = np.array(metric_name)
    labels = np.concatenate((labels,[labels[0]]))

    nAttr = len(metric_name)

    # print(metric_name)
    # print(api_scores[0])
    # print(api_scores[1])

    angles = np.linspace(0+np.pi/2, np.pi/2+2*np.pi, nAttr, endpoint=False)
    angles = np.concatenate((angles,[angles[0]]))
    
    fig = plt.figure(facecolor="white")
    ax = plt.subplot(111,polar=True)
    ax.set_ylim(0,5)

    api_color_map = {}
    for i, api_name in enumerate(api_names):
        api_color_map[api_name] = colors[int(i%len(colors))]

    for i, api_name in enumerate(api_names):
        if i == 0:
            plt.thetagrids(angles*180/np.pi,labels)

        api_score = api_name_score[api_name]
        data = np.array(api_score)
        data = np.concatenate((data,[data[0]]))

        ## show api name
        # plt.plot(angles,data,'-',color=api_color_map[api_name],linewidth=1,alpha=0.5,label=api_name)
        # plt.fill(angles,data,facecolor=api_color_map[api_name],alpha=0.05)
        plt.plot(angles,data,'-',linewidth=1,alpha=0.5,label=api_name)
        plt.fill(angles,data,alpha=0.05)

    ## show title
    plt.figtext(0.515,0.95,attr,ha='center')

    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"plots/Figures/{attr}.png",dpi=600)