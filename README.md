### Step One: deploy the large language model.
Just run the following script, where num_server means the api number.
```
sh server_deploy.sh --num_server 16
```

Note: the api server information will be recorded in 'copywriting/urls/{model_name}_server_info.txt'.

### Step Two: complete human evaluation tasks with the deployed large language model.
```
python llm_generation.py --model_name chatglm --batch_size 4
```

Note: make sure that the api servers in 'copywriting/urls/{model_name}_server_info.txt' is already.