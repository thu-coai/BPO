from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# TODO change model path
model_path = 'Your-Model-Path'

prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"

model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    

def gen(input_text):
    prompt = prompt_template.format(input_text)
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
    resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()

    print("[Stable Optimization] ", resp)


def gen_aggressive(input_text):
    texts = [input_text] * 5  
    responses = []
    for text in texts:
        seed = torch.seed()
        torch.manual_seed(seed)
        prompt = prompt_template.format(text)
        min_length = len(tokenizer(prompt)['input_ids']) + len(tokenizer(text)['input_ids']) + 5
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        bad_words_ids = [tokenizer(bad_word, add_special_tokens=False).input_ids for bad_word in ["[PROTECT]", "\n\n[PROTECT]", "[KEEP", "[INSTRUCTION]"]]
        # eos and \n
        eos_token_ids = [tokenizer.eos_token_id, 13]
        output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.9, bad_words_ids=bad_words_ids, num_beams=1, eos_token_id=eos_token_ids, min_length=min_length)
        resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].split('[KE')[0].split('[INS')[0].split('[PRO')[0].strip()
        responses.append(resp)

    for i in responses:
        print("[Aggressive Optimization] ", i)


text = 'how can I create a profile on Facebook?'

# Stable optimization, this will sometimes maintain the original prompt
gen(text)

# Agressive optimization, this will refine the original prompt with a higher possibility
# but there may be inappropriate changes
gen_aggressive(text)
