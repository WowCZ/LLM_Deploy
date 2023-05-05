import json

def process_humor(data_file: str) -> list:
    with open(data_file, 'r') as fr:
        humors = fr.readlines()
        humor_sentences = []
        for hl in humors:
            hl = hl.split('.')[-1].strip()

            if len(hl) <= 3:
                continue

            if hl not in humor_sentences:
                humor_sentences.append(hl)
    
    prompt_temp = "有这样一个故事，“{story}”，请问这个故事的笑点在哪儿？"

    humor_sentences = [{'humor_sentence': hl, 'prompt_template': prompt_temp, 'prompt': prompt_temp.format(story=hl)} for hl in humor_sentences]

    out_data_file = data_file.replace('_origin', '').replace('.txt', '.json')
    with open(out_data_file, 'w') as fw:
        json.dump(humor_sentences, fw, indent=4, ensure_ascii=False)

    return humor_sentences

def process_default_task(data_file: str) -> list:
    poetry_instruction = []
    with open(data_file, 'r') as fr:
        poetry = fr.readlines()
        for p in poetry:
            poetry_instruction.append(p.strip())
    
    poetry_instruction = [{'prompt': p} for p in poetry_instruction]

    out_data_file = data_file.replace('_instruction', '').replace('.txt', '.json')
    with open(out_data_file, 'w') as fw:
        json.dump(poetry_instruction, fw, indent=4, ensure_ascii=False)
    
    return poetry_instruction


if __name__ == '__main__':
    humor_file = 'copywriting/data/humor_origin.txt'
    humor_sentences = process_humor(humor_file)
    print(humor_sentences[0])
    print(humor_sentences[-1])
    print(len(humor_sentences))

    story_file = 'copywriting/data/story_instruction.txt'
    stories = process_default_task(story_file)
    print(stories[0])
    print(stories[-1])
    print(len(stories))

    poetry_file = 'copywriting/data/poetry_instruction.txt'
    poetries = process_default_task(poetry_file)
    print(poetries[0])
    print(poetries[-1])
    print(len(poetries))