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
from anthropic import Anthropic

anthropic = Anthropic(
    api_key="Your-API-Key",
)


def claude_gen(messages, counter, error_count):
    responses = []
    for i, m in enumerate(messages):
        try:
            message = m['message']
            completion = anthropic.completions.create(
                model='claude-v1.3',
                max_tokens_to_sample=512,
                prompt=f"{message}",
                temperature=0.0
            )
            print(completion)
            resp = completion.completion
            m['response'] = resp
            # save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(resp)

            # Increment and print the counter
            counter += 1
        except Exception as e:
            error_count += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time() - s_time, counter,
                                                                            error_count), end='\r')

    return responses


def get_messages_list():
    if task_name.count("test_set") or task_name.count("dolly"):
        idx = "idx"
    elif task_name.count("self_instruct"):
        idx = "id"
    elif task_name.count("vicuna"):
        idx = "question_id"
    else:
        print("Not implemented")
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

    for i, j in zip(d_a, d_b):
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
            question = (i['instruction'] + '\n' + i['context']).strip()
        elif task_name.count("test_set"):
            question = i['context'].strip()
        elif task_name.count("vicuna"):
            question = i['text'].strip()
        else:
            print("Not implemented")
            assert False
        messages_list.append({'message': prompt.replace('{instruction}', question).replace('{output_1}', res_a).replace(
            '{output_2}', res_b),
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

    with open('./evaluation/ranking_prompt.txt') as f:
        lines = f.readlines()
        prompt = ''
        for i in lines:
            prompt = prompt + i
    if not os.path.exists(output_file):
        x = open(output_file, 'w')
        x.close()
    messages_list = get_messages_list()
    print("total num: ", len(messages_list))
    s_time = time.time()
    responses = claude_gen(messages_list, 0, 0)
