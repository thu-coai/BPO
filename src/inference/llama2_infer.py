from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
import time
from collections import OrderedDict

device = 'cuda:0'


model_name = "Llama-2-7b-chat-hf"
prompt_template = "[INST] {} [/INST]"


model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# BPO-optimized prompts 
with open('dolly_eval_optimized.json') as f:
    data = json.load(f)


with torch.no_grad():
    res = []
    for i in tqdm(data):
        input_text = prompt_template.format((i['optimized_prompt']).strip())
        model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        output = model.generate(**model_inputs, max_new_tokens=2048, do_sample=True, top_p=1.0, temperature=0.7)
        resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
        i['res'] = resp
        res.append(i)

with open('dolly_eval_optimized_llama2_7b_res.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
