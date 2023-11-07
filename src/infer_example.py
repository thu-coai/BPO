from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'Your-Model-Path'

prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"

model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = 'Tell Me About Harry Potter'

prompt = prompt_template.format(text)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.9, num_beams=1)
resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()

print(resp)
