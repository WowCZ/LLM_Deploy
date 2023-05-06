from glob import glob
import json
import argparse
import shutil


parser = argparse.ArgumentParser(description='Set all the prompts to be the default prompts.')
parser.add_argument('--task', type=str, default='empathy', help='human evaluation task')
args = parser.parse_args()
task = args.task

prompt_folder = 'copywriting/data/'
task_file = f'{prompt_folder}{task}.json'
result_files = glob(f'{prompt_folder}{task}_*.json')
if f'{prompt_folder}{task}_to_submit.json' in result_files:
    result_files.remove(f'{prompt_folder}{task}_to_submit.json')

with open(task_file) as f:
    original_prompts = json.load(f)
prompts_to_submit = dict()
for i in range(len(original_prompts)):
    prompts_to_submit[f"prompt{i + 1}"] = original_prompts[i]["prompt"]
with open(task_file[: task_file.index(".json")] + "_to_submit.json", "w", encoding="utf-8") as f:
    json.dump(prompts_to_submit, f, indent=4, ensure_ascii=False)

for result_file in result_files:
    with open(result_file) as f:
        result = json.load(f)
    assert len(original_prompts) == len(result), f"The number of prompts are not the same! {task_file} & {result_file}"
    for i in range(len(original_prompts)):
        result[i]["prompt"] = original_prompts[i]["prompt"]
    # old_name = result_file[: result_file.index(".json")] + "_old.json"
    # shutil.copyfile(result_file, old_name)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


