import os
import emoji
import time
import tqdm
import json
import gradio as gr
from copywriting import sample_trueskill
from llm_api import ability_name_map, api_name_map, ability_en_zh_map

abilities = [k for k in ability_name_map.keys()]
llms = [k for k in api_name_map.values()]
api_name_reverse_map = dict([(v, k) for k, v in api_name_map.items()])
ability_zh_en_map = dict([(v, k) for k, v in ability_en_zh_map.items()])

class ArenaPlan():
    def __init__(self, 
                 match_plan: list, 
                 ability: str, 
                 sample_num: int=1, 
                 inference_path: str='copywriting/data', 
                 dump_save_path: str='copywriting/analysis/arena', 
                 dump_recovery_path: str='copywriting/analysis/arena'):
        self.matches = sample_trueskill(match_plan, ability, sample_num, inference_path, dump_save_path, dump_recovery_path)
        self.match_cnt = len(self.matches)
        self.annotating_head = 0
        self.show_head = 0
        self.annotated_matches = []
        self.dump_save_path = dump_save_path
        self.ability = ability

    def _show_annotating_match(self, match_id: int):
        match = self.matches[match_id]
        for game_name, match_content in match.items():
            instruct = match_content['指令任务1']
            model_a = match_content['模型回复1']['模型1']
            model_b = match_content['模型回复1']['模型2']
            return game_name, instruct, model_a, model_b, None
        
    def _show_annotated_match(self, match_id: int):
        match = self.annotated_matches[match_id]
        for game_name, match_content in match.items():
            instruct = match_content['instruction']
            model_a = match_content['model-a']
            model_b = match_content['model-b']
            winner = match_content['winner']
            return game_name, instruct, model_a, model_b, winner
        
    def show_match(self, match_id: int =0):
        match_id = max(0, match_id)
        match_id = min(match_id, self.match_cnt-1)

        if match_id < self.annotating_head:
            return self._show_annotated_match(match_id)
        else:
            return self._show_annotating_match(match_id)

    def next_match(self):
        self.show_head += 1
        self.show_head = min(self.show_head, self.match_cnt-1)

        return self.show_match(self.show_head)
    
    def last_match(self):
        self.show_head -= 1
        self.show_head = max(0, self.show_head)

        return self.show_match(self.show_head)
    
    def progress(self):
        return f'{self.annotating_head}/{self.match_cnt}'
    
    def submit_match(self, match_result: dict):
        game_name = list(self.matches[self.show_head].keys())[0]
        self.annotated_matches.append({game_name: match_result})
        if self.show_head == self.annotating_head:
            self.annotating_head += 1
    
    def save_matches(self):
        save_match_file = os.path.join(self.dump_save_path, f'{self.ability}.json')
        with open(save_match_file, 'w') as fw:
            json.dump(self.annotated_matches, fw, ensure_ascii=False, indent=4)

def full_combination_strategy(llms):
    match_plan = []
    llm_cnt = len(llms)
    for li in range(llm_cnt):
        for ci in range(li+1, llm_cnt):
            match_plan.append(f'{llms[li]}&{llms[ci]}')

    return match_plan

def submit(instruct, model_a, model_b, winner):
    global arena_plan
    submit_result = {
        'instruction': instruct,
        'model-a': model_a,
        'model-b': model_b,
        'winner': winner
    }
    arena_plan.submit_match(submit_result)
    return f'Winner 🎖 is {winner}! \n ({arena_plan.progress()})'

def next_match():
    global arena_plan
    game_name, instruct, model_a, model_b, winner = arena_plan.next_match()
    return instruct, model_a, model_b, winner

def last_match():
    global arena_plan
    game_name, instruct, model_a, model_b, winner = arena_plan.last_match()
    return instruct, model_a, model_b, winner

def load_match(ability, llms):
    global arena_plan
    llms = [api_name_reverse_map[l] for l in llms]

    if len(llms) < 2:
        return 'please select more than one language models in the pool!'
    
    match_plan = full_combination_strategy(llms)
    print(match_plan)
    arena_plan = ArenaPlan(match_plan, ability_zh_en_map[ability])
    game_name, instruct, model_a, model_b, winner = arena_plan.show_match()

    return f'{len(match_plan)} battles have been loaded!', instruct, model_a, model_b, winner

def save_result():
    global arena_plan
    arena_plan.save_matches()

def arena_two_model():
    with gr.Blocks(title=emoji.emojize('Arena :crossed_swords:')) as demo:
        gr.Markdown('**Step 1**: Choose the ability for evaluation:')
        human_eval_abi = gr.Dropdown(choices=[ability_name_map[k] for k in abilities], 
                                     label=emoji.emojize(f'{len(abilities)} abilities for human evaluation :technologist:'))
        
        gr.Markdown('**Step 2**: Choose the large language models for battle:')
        llm_pool = gr.CheckboxGroup(choices=llms,
                                    label=emoji.emojize(f'{len(abilities)} large language models in the zoo :paw_prints:'))
        
        gr.Markdown('**Step 3**: Load the battle matches:')
        with gr.Row():
            match_plan_btn = gr.Button(emoji.emojize('Loading Match Plan 👉'))
            plan_progress = gr.Label(label='Loading Progress ⌛')

        gr.Markdown('**Step 4**: Make your decision about the following battle 🎮:')
        instruction = gr.Textbox(label=emoji.emojize('Instruction :person_raising_hand:'))
        with gr.Row():
            response_a = gr.Textbox(label='Responses from Model-A 🔴', lines=8)
            response_b = gr.Textbox(label='Responses from Model-B 🟢', lines=8)

        with gr.Row():
            radio = gr.Radio(['Model-A 🔴', 'Model-B 🟢'], label='Competitor')
            result = gr.Label(label='Winner 🎖')

        with gr.Row():
            last_btn = gr.Button(emoji.emojize('Last Match :last_track_button:'))
            next_btn = gr.Button(emoji.emojize('Next Match :next_track_button:'))
            submit_btn = gr.Button(emoji.emojize('Submit :thumbs_up:'))
            save_btn = gr.Button(emoji.emojize('Save 📥'))

        submit_btn.click(fn=submit, 
                         inputs=[instruction, response_a, response_b, radio], 
                         outputs=result, 
                         api_name='submit')
        
        next_btn.click(fn=next_match, 
                       inputs=None, 
                       outputs=[instruction, response_a, response_b, radio], 
                       api_name='next')
        
        last_btn.click(fn=last_match, 
                       inputs=None, 
                       outputs=[instruction, response_a, response_b, radio], 
                       api_name='last')
        
        match_plan_btn.click(fn=load_match, 
                       inputs=[human_eval_abi, llm_pool], 
                       outputs=[plan_progress, instruction, response_a, response_b, radio], 
                       api_name='match_plan')
        
        save_btn.click(fn=save_result, 
                       inputs=None, 
                       outputs=None, 
                       api_name='save')

    demo.launch(share=True)