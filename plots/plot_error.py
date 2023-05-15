import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from human_eval_preprocess import file_reader
# matplotlib.rcParams['font.family']='SimHei'
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns
sns.set_theme(style="whitegrid")

def errplot(x, y, yerr, hue, **kwargs):
    data = kwargs.pop('data')
    p = data.pivot_table(index=x, columns=hue, values=y, aggfunc='mean')
    err = data.pivot_table(index=x, columns=hue, values=yerr, aggfunc='mean')
    p.plot(kind='bar', yerr=err, ax=plt.gca(), **kwargs)

df, analysis_results = file_reader('plots/HumanEvalResults/EvalPlat_subjective/安全能力')

bar_df = {
    'metric': [],
    'score_mean': [],
    'score_std': [],
    'api_name': []
}
for t_name, a_data in analysis_results.items():
    for a_name, m_data in a_data.items():
        bar_df['metric'].extend(m_data['metric'])
        bar_df['score_mean'].extend(m_data['score_mean'])
        bar_df['score_std'].extend(m_data['score_std'])
        bar_df['api_name'].extend(m_data['api_name'])

bar_df = pd.DataFrame(bar_df)
bar_df.to_csv(f'plots/HumanEvalResults/Statistics/{t_name}.csv')


g = sns.FacetGrid(bar_df)
g.map_dataframe(errplot, "metric", "score_mean", "score_std", "api_name", width=0.8)

# plt.subplots_adjust(right=0.90)
plt.legend(loc='center left', bbox_to_anchor=(1,1))

# # Draw a nested barplot by species and sex
# g = sns.catplot(
#     data=bar_df, kind="bar",
#     x="metric", y="score_mean", hue="api_name",
#     errorbar="sd", palette="dark", alpha=.6, height=6
# )
# g.despine(left=True)
# g.set_axis_labels("", "Human Evaluation Score")
# g.legend.set_title("LLM")

plt.savefig(f"plots/Figures/{t_name}.png",dpi=600)

