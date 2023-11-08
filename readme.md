
<!-- <div align="center"> -->
<!-- <img src="assets/cover.png" alt="BPO" width="90%" /> -->
<!-- </div> -->
# Black-Box Prompt Optimization (BPO)
### Aligning Large Language Models without Model Training

<p align="center">
   🌐 <a href="#model" target="_blank">Model</a> • ⏬ <a href="#data" target="_blank">Data</a> • 📃 <a href="https://arxiv.org/abs/2311.04155" target="_blank">Paper</a>
</p>

(Upper) Black-box Prompt Optimization (BPO) offers a conceptually new perspective to bridge the gap between humans and LLMs. (Lower) On Vicuna Eval’s pairwise evaluation, we show that BPO further aligns gpt-3.5-turbo and claude-2 without training. It also outperforms both PPO & DPO and presents orthogonal improvements.

<div align="center">
<img src="assets/intro.png" alt="BPO" width="50%" />
</div>

<br>
<br>

## Table of Contents
- [Model](#model)
- [Data](#data)
- [Quick Start](#quick-start)
    - [Data Construction](#data-construction)
    - [Model Training](#model-training)
- [Citation](#citation)


## Model
The prompt preference optimization model can be download from [BPO Model](https://cloud.tsinghua.edu.cn/d/d2af23156eab4c148672/)

Inference code (Please refer to [src/infer_example.py](src/infer_example.py) for more instructions on how to optimize your prompts):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = 'Your-Model-Path'

prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"

model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = 'Tell me about Harry Potter'

prompt = prompt_template.format(text)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
output = model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()

print(resp)
```

## Data

### BPO dataset
We will open source our data after further checking.


### BPO for SFT Data Construction 
The alpaca_reproduce directory contains the BPO-reproduced Alpaca dataset. The data format is:
```json
{
    "instruction": {instruction},
    "input": {input},
    "output": {output},
    "optimized_prompt": {optimized_prompt},
    "res": {res}
}
```
- {instruction}, {input}, and {output} are elements from the original dataset.
- {optimized_prompt} is BPO-optimized instruction.
- {res} is the response from text-davinci-003 using the {optimized_prompt}.


### Testset
The testset directory contains all the test datasets we used, including: 
- 200 prompts sampled from the BPO dataset
- 200 examples from Dolly dataset
- 252 human evaluation instructions from Self-Instruct
- 80 user-oriented prompts from the Vicuna Eval dataset.


## Quick Start
For all codes, we have added `#TODO` comments to indicate places in the code that need modification before running. Please update the relevant parts as noted before executing each file.

### Setup
```bash
pip install -r requirements.txt
```

### Data Construction
To construct data yourself, run the following command
```bash
cd src/data_construction

# using pairwise feedback data to generate optimized prompts
python chatgpt_infer.py

# process generated optimized prompts
python process_optimized_prompts.py
```

### Model Training
If you want to train your own prompt preference optimizer, 
please run the following command:
```bash
cd src/training

# pre-process fine-tuning data
python ../data_construction/process_en.py
python data_utils.py

# fine-tuning
python train.py

# inference
python infer_finetuning.py
```

## TODO
- [ ] Datasets
- [ ] Inference Code
- [ ] Evaluation Code
- [ ] RLHF Code

## Acknowledgement
- Fine-tuning code: [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- PPO code: [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md)
- DPO code: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Evaluation Prompts: [llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) and [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)

## Citation
```
@misc{cheng2023blackbox,
      title={Black-Box Prompt Optimization: Aligning Large Language Models without Model Training}, 
      author={Jiale Cheng and Xiao Liu and Kehan Zheng and Pei Ke and Hongning Wang and Yuxiao Dong and Jie Tang and Minlie Huang},
      year={2023},
      eprint={2311.04155},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
