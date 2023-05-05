### Step One: deploy the large language model.
Just run the following script, where num_server means the api number.
```
sh server_deploy.sh --num_server 16
```

Note: the api server information will be recorded in 'copywriting/urls/{model_name}_server_info.txt'.

### Step Two: complete human evaluation tasks with the deployed large language model apis.
```
python api_visit.py --model_name davinci --batch_size 2 --max_length 1024
```

Note: make sure that the api servers in 'copywriting/urls/{model_name}_server_info.txt' is already existed.

### Step Three: simulate as the human evaluation server.
```
python simulator.py --model_name davinci --task empathy --api_port 6566
```