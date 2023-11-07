import json
import random
from tqdm import trange, tqdm
import os

# Preprocess code for dataset with context, like Alpaca-gpt4
def process_no_ctx(input_file, output_file):

    with open(input_file) as f:
        l = f.readlines()

    res = []
    for i in l:
        i = json.loads(i)
        response = i['response'].split('[END]')[0]
        if not response.count('Optimized Instruction:'):
            print(response)
            continue
        else:
            response = response.split('Optimized Instruction:')
        try:
            prompt = eval(response[1]).strip()
        except:
            prompt = response[1].strip()
        i['origin']['comparison'] = response[0]
        i['origin']['optimized_prompt'] = prompt
        res.append(i['origin'])


    data = []
    for i in tqdm(res):
        if not len(i['context']):
            i['final_input'] = i['instruction']
            i['final_output'] = i['optimized_prompt']
        else:
            # optimized instruction contains context
            if i['optimized_prompt'].lower().count(i['context'].lower()):
                i['final_input'] = (i['instruction'] + '\n' + i['context']).strip()
                i['final_output'] = i['optimized_prompt']
            else:
                # using the format {instruction}\n{context}
                if i['optimized_prompt'].count('follow') or i['instruction'].count('follow') or i['instruction'][-1] == ':':
                    i['final_input'] = (i['instruction'] + '\n' + i['context']).strip()
                    i['final_output'] = (i['optimized_prompt'] + '\n' + i['context']).strip()
                else:
                    if random.random()< 0.5:
                        if random.random() < 0.5:
                            # using the format {instruction}\n{context}
                            i['final_input'] = (i['instruction'] + '\n' + i['context']).strip()
                            i['final_output'] = (i['optimized_prompt'] + '\n' + i['context']).strip()
                        else:
                            # using the format {context}\n{instruction}
                            i['final_input'] = (i['context'] + '\n' + i['instruction']).strip()
                            i['final_output'] = (i['context'] + '\n' + i['optimized_prompt']).strip()
                    else:
                        if random.random() < 0.25:
                            if random.random() < 0.5:
                                # using the format {instruction} {context}
                                i['final_input'] = (i['instruction'] + ' ' + i['context']).strip()
                                i['final_output'] = (i['optimized_prompt'] + ' ' + i['context']).strip()
                            else:
                                # using the format {context} {instruction}
                                i['final_input'] = (i['context'] + ' ' + i['instruction']).strip()
                                i['final_output'] = (i['context'] + ' ' + i['optimized_prompt']).strip()
                        else:
                            if random.random() < 0.5:
                                # using the format {instruction} "{context}"
                                i['final_input'] = (i['instruction'] + ' "' + i['context'] + '"').strip()
                                i['final_output'] = (i['optimized_prompt'] + ' "' + i['context'] + '"').strip()
                            else:
                                # using the format {context} "{instruction}"
                                i['final_input'] = (i['context'] + ' "' + i['instruction'] + '"').strip()
                                i['final_output'] = (i['context'] + ' "' + i['optimized_prompt'] + '"').strip()
        data.append(i)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Preprocess code for dataset without context, like Chatbot Arena Conversation
def process_ctx(input_file, output_file):

    with open(input_file) as f:
        l = f.readlines()

    res = []
    for i in l:
        i = json.loads(i)
        response = i['response'].split('[END]')[0]
        if not response.count('Optimized Instruction:'):
            print(response)
            continue
        else:
            response = response.split('Optimized Instruction:')
        try:
            prompt = eval(response[1]).strip()
        except:
            prompt = response[1].strip()
        i['origin']['comparison'] = response[0]
        i['origin']['optimized_prompt'] = prompt
        res.append(i['origin'])

    data = []
    num = 0
    for i in res:
        if len(i['prompt'].split()) / len(i['optimized_prompt'].split()) > 2:
            # filter data that may miss some contents
            continue
        if i['optimized_prompt'].lower().count('[protected'):
            # filter data contains special string
            continue
        i['final_input'] = i['instruction']
        i['final_output'] = i['optimized_prompt']
        data.append(i)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # TODO add input_file output_file
    input_file = '../data/data_construction/examples_ctx_optimized.jsonl'
    output_file = '../data/data_construction/examples_ctx_optimized.json'
    
    # TODO choose a function depend on your dataset

    # Preprocess code for dataset with context attribute, like Alpaca-gpt4
    process_ctx(input_file, output_file)

    # Preprocess code for dataset without context attribute, like Chatbot Arena Conversation
    # process_no_ctx(input_file, output_file)