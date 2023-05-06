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
if f'{prompt_folder}{task}_answer.json' in result_files:
    result_files.remove(f'{prompt_folder}{task}_answer.json')

with open(task_file) as f:
    original_prompts = json.load(f)

if task == "reading":
    with open(f'{prompt_folder}{task}_answer.json') as f:
        raw = json.load(f)
    # Reading 校验已通过，问题与raw对得上
    # assert len(raw) == len(original_prompts)
    # for i in range(len(original_prompts)):
    #
    #     prompt = original_prompts[i]["prompt"]
    #     question = prompt[prompt.index("\n问题：") + 4:].strip()
    #     if question != raw[i]["question"]:
    #         print(f"{i}\t{question}\t{raw[i]['question']}")

prompts_to_submit = dict()
for i in range(len(original_prompts)):
    prompts_to_submit[f"prompt{i + 1}"] = original_prompts[i]["prompt"] if task != "reading" else f"{original_prompts[i]['prompt']}参考答案（仍需阅读原文）：{raw[i]['answer']}\n"
with open(task_file[: task_file.index(".json")] + "_to_submit.json", "w", encoding="utf-8") as f:
    json.dump(prompts_to_submit, f, indent=4, ensure_ascii=False)

for result_file in result_files:
    with open(result_file) as f:
        result = json.load(f)
    assert len(original_prompts) == len(result), f"The number of prompts are not the same! {task_file} & {result_file}"
    for i in range(len(original_prompts)):
        result[i]["prompt"] = original_prompts[i]["prompt"] if task != "reading" else f"{original_prompts[i]['prompt']}参考答案（仍需阅读原文）：{raw[i]['answer']}\n"
    # old_name = result_file[: result_file.index(".json")] + "_old.json"
    # shutil.copyfile(result_file, old_name)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


