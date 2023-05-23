## Inference process with API server, API client and API simulator.

### Step 1: deploy the large language model.
Just run the following script, where num_server means the api number.

Choice 1: refer to: scripts/api_server.py
```
python api.py server --api=T5API --wrapper=Flask
```
Choice 2: asynchronous task submission
```
sh server_deploy.sh --num_server 16
```

Note: the api server information will be recorded in 'copywriting/urls/{model_name}_server_info.txt'.

### Step 2: complete human evaluation tasks with the deployed large language model apis.
```
python api.py client --model_name davinci --batch_size 2 --max_length 1024
```

Note: make sure that the api servers in 'copywriting/urls/{model_name}_server_info.txt' is already existed.

### Step 3: simulate as the human evaluation server.
```
python api.py simulator --model_name davinci --simulate_task empathy --port 6566
```

## Analysis process with annotated results.

### To analysis the annotated results at first stage:
```
python analysis.py recovery --name human_evaluation --annotated_path copywriting/annotated_data/human_evaluation --dump_result_path copywriting/annotated_data/analysis_data --save_fig_path plots/figures
```

### To sample annotating data for TrueSkill strategy:
```
python analysis.py sampling --name trueskill_evaluation --match_plan 'alpaca&belle' 'alpaca&bloom' --single_sample_size 3 --evaluation_tasks 'empathy' --dump_recovery_path copywriting/annotated/trueskill_recovery --annotating_path copywriting/annotated/trueskill
```

### To recover annotated data for TrueSkill strategy:
```
python analysis.py recovery --name trueskill_evaluation --recovery_tasks 'empathy' --annotated_path copywriting/annotated/trueskill_recovery --annotating_path copywriting/annotated/trueskill --dump_result_path copywriting/annotated/analysis_data
```

### To plot figures:
```
python analysis.py plot --type gaussian --data_file copywriting/annotated_data/trueskill --save_fig_path plots/figures/gaussian --save_fig_name dynamic
```