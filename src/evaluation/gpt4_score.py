import requests
import multiprocessing
from multiprocessing import Manager
import json
from tqdm import tqdm
import os
import time
import pandas as pd
import random
import argparse

API_KEY = 'Your-API-Key'

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

API_URL = "https://api.openai.com/v1/chat/completions"

def chat_gpt(messages, counter, error_count):
    responses = []
    for i, m in enumerate(messages):
        try:
            message = m['message']
            data = json.dumps({"model": "gpt-4", "messages": message, 'temperature': 0.0})
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            print(response_json)
            res = response_json['choices'][0]['message']['content']
            m['response'] = res
            # save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(response_json)

            # Increment and print the counter
            counter += 1
        except Exception as e:
            error_count += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter, error_count), end='\r')

    return responses


def get_messages_list():

    if task_name.count("test_set") or task_name.count("dolly"):
        idx = "idx"
    elif task_name.count("self_instruct"):
        idx = "id"
    elif task_name.count("vicuna"):
        idx = "question_id"
    else:
        print("idx Not implemented")
        assert False

    evaluated = []
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()
    for i in lines:
        evaluated.append(json.loads(i)['origin'])

    with open(input_file_a) as f:
        d_a = json.load(f)

    with open(input_file_b) as f:
        d_b = json.load(f)

    messages_list = []

    for i,j in zip(d_a, d_b):
        assert (i[idx] == j[idx])
        if i[idx] in evaluated:
            continue
        if random.randint(0, 1) == 0:
            option_a = i
            res_a = i['res']
            res_b = j['res']
        else:
            option_a = j
            res_a = j['res']
            res_b = i['res']
        if task_name.count("self_instruct") or task_name.count("dolly"):
            question = (i['instruction']+'\n'+i['context']).strip()
        elif task_name.count("test_set"):
            question = i['context'].strip()
        elif task_name.count("vicuna"):
            question = i['text'].strip()
        else:
            print("Not implemented")
            assert False
        messages_list.append({'message': [
                {"role": 'system', "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['prompt_template'].replace('{question}', question).replace('{answer_a}', res_a).replace('{answer_b}', res_b)}
            ],
            'origin': i[idx],
            'option_a': option_a,
        })
        
    return messages_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file_a', type=str)
    parser.add_argument('--input_file_b', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()  
    
    input_file_a = args.input_file_a
    input_file_b = args.input_file_b
    task_name = args.task_name
    output_file = args.output_file

    prompt = {"name": "pair-v2", "type": "pairwise", "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]"}
    if not os.path.exists(output_file):
        x = open(output_file, 'w')
        x.close()
    messages_list = get_messages_list(task_name)
    print("total num: ", len(messages_list))
    s_time = time.time()
    responses = chat_gpt(messages_list, 0, 0)
