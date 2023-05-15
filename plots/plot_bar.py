import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from human_eval_preprocess import file_reader
# matplotlib.rcParams['font.family']='SimHei'
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
sns.set_theme(style="whitegrid")

# penguins = sns.load_dataset("penguins")
sns.palplot(sns.color_palette("hls", 12))

df, analysis_results = file_reader('plots/HumanEvalResults/EvalPlat_subjective/幽默理解')

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
print(hue_order)
bar_df = pd.DataFrame(bar_df)
bar_df.sort_values(by='metric', inplace=True, ascending=True)
bar_df.to_csv(f'plots/HumanEvalResults/Statistics/{t_name}.csv')

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
plt.savefig(f"plots/Figures/{t_name}.png",dpi=600)

