# 👽ALIEN: Ability Leading Evaluation 

## 🚀LLM APIs 
-------
### Deploy the large language model:
```
python api.py server --api=T5API --wrapper=Flask
```

### Complete human evaluation tasks with the deployed large language model apis:
```
python api.py client --model_name davinci --batch_size 2 --max_length 1024
```

### Simulate as the human evaluation server:
```
python api.py simulator --model_name davinci --simulate_task empathy --port 6566
```

## 📈Result Analysis 
-------
### Sample annotating data for TrueSkill strategy:
```
python analysis.py sampling \
 --name trueskill_evaluation \
 --match_plan 'alpaca&belle' 'alpaca&bloom' \
 --sample_num 3 \
 --evaluation_tasks 'empathy' \
 --dump_recovery_path resource/annotated/trueskill_recovery \
 --annotating_path resource/annotated/trueskill
```

### Recover annotated data for TrueSkill strategy:
```
python analysis.py recovery \
 --name trueskill_evaluation \
 --recovery_tasks 'empathy' \
 --annotated_data_path resource/annotated/trueskill \
 --recovery_info_path resource/annotated/trueskill_recovery \
 --dump_result_path resource/annotated/analysis_data
```

### Plot figures:
```
python analysis.py plot \
 --type gaussian \
 --data_file resource/annotated_data/trueskill \
 --save_fig_path resource/figures/gaussian \
 --save_fig_name gaussian
```

## 🌐WebUI 
-------
```
python webui.py
```
![image](assets/figures/trueskill_annotation.png)
