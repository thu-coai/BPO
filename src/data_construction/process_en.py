import json
import random

with open('../../data/data_construction/examples_ctx_optimized.json', encoding='utf-8') as f:
    d = json.load(f)

res = []
num = 0
for i in d:
    q = i['prompt']
    a = i['optimized_prompt']
    try:
        a = eval(a)
    except:
        pass
    res.append(json.dumps({
        'id': num,
        "paragraph": [
            {
                'q': q,
                'a': a
            }
        ],
    }, ensure_ascii=False) + '\n')
    num += 1

with open('data/train.jsonl', 'w', encoding='utf-8') as f:
    f.writelines(res)

