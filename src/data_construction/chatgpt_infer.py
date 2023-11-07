import requests
import json
import os
import time
import random

# TODO add api key
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
            data = json.dumps({"model": "gpt-3.5-turbo", "messages": message, 'temperature':0.9})
            response = requests.post(API_URL, headers=HEADERS, data=data)
            response_json = response.json()
            res = response_json['choices'][0]['message']['content']
            m['response'] = res
            # save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(response_json)

            counter += 1
        except Exception as e:
            error_count += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter, error_count), end='\r')

    return responses


def get_messages_list():
    evaluated = []
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()
    for i in lines:
        evaluated.append(json.loads(i)['origin'])

    with open(input_file, encoding='utf-8') as f:
        d = json.load(f)
    
    messages_list = []
    
    ctx_prompt = """instruction: "{}"
context:
"{}"

bad response:
"{}"

good response:
"{}"

Compare the good response and bad response from these aspects: correctness (if the response follows the instruction correctly and give an accurate response, high priority), helpfulness(like depth, creativity, coherence) and harmlessness. Then be an expert prompt engineer and improve my instruction from the above aspects to get better responses like "good response" rather than "bad response". 

Pay attention to:
1.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.
2.Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.
3.Don't change the context or add the context into the instruction, but rather optimize my instruction only. Don't give a response to my instruction.
4.Help me tune my prompt (the instruction) to get a better response while remaining the original meaning of the instruction and user intent.

Output with the following format:
Detailed Comparison Result: xxx
Optimized Instruction: xxx [END]"""

    no_ctx_prompt = """instruction: "{}"

bad response:
"{}"

good response:
"{}"

Compare the good response and bad response from these aspects: correctness (if the response follows the instruction correctly and give an accurate response, high priority), helpfulness(like depth, creativity, coherence) and harmlessness. Then be an expert prompt engineer and improve my instruction from the above aspects to get better responses like "good response" rather than "bad response". 

Pay attention to:
1.If the instruction contains any safety issues, please rewrite the original instructions to be completely harmless and safe under the same topic.
2.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.
3.Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.
4.There may be some protected parts in the instruction, which means these parts should never be changed or lost. Please carefully protect these parts.
5.You should never generate a response to the original instruction!
6.Help me tune my prompt (the instruction) to get a better response while maintaining the original meaning of the instruction and the user intent.

Output with the following format:
Detailed Comparison Result: xxx
Optimized Instruction: xxx [END]"""

    for i in d:
        if i in evaluated:
            continue
        if 'context' in i:
            text = ctx_prompt.format(i['instruction'], i['context'], i['bad_res'], i['good_res'])
        else:
            text = no_ctx_prompt.format(i['instruction'], i['bad_res'], i['good_res'])
        messages_list.append({
            'message': [
                {"role": "user", "content": text}
            ],
            'origin': i
        })
        
    return messages_list


if __name__ == '__main__':
    # TODO input file and output file
    input_file = '../data/data_construction/examples_ctx.json'
    output_file = '../data/data_construction/examples_ctx_optimized.jsonl'
    if not os.path.exists(output_file):
        x = open(output_file, 'w')
        x.close()
    messages_list = get_messages_list()
    print("total num: ", len(messages_list))
    s_time = time.time()
    responses = chat_gpt(messages_list, 0, 0)