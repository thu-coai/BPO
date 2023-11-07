
<!-- <div align="center"> -->
<!-- <img src="assets/cover.png" alt="BPO" width="90%" /> -->
<!-- </div> -->
# Black-Box Prompt Optimization (BPO)
### Aligning Large Language Models without Model Training

<p align="center">
   <!-- •   📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a> -->
</p>

(Upper) Black-box Prompt Optimization (BPO) offers a conceptually new perspective to bridge the gap between humans and LLMs. (Lower) On Vicuna Eval’s pairwise evaluation, we show that BPO further aligns gpt-3.5-turbo and claude-2 without training. It also outperforms both PPO & DPO and presents orthogonal improvements.

<!-- ![BPO](assets/intro.png) -->
<div align="center">
<img src="assets/intro.png" alt="BPO" width="50%" />
</div>

<br>
<br>

## Table of Contents
- [Quick Start](#quick-start)
    - [Data Construction](#data-construction)
    - [Model Training](#model-training)
<!-- - [Citation](#citation) -->


## Model & Data
We will open source the data and prompt optimization model once they are organized.


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
- [ ] Prompt Optimization Model
- [ ] Inference Code
- [ ] Evaluation Code
- [ ] RLHF Code

## Acknowledgement
- Fine-tuning code: [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- PPO code: [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md)
- DPO code: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Evaluation Prompts: [llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) and [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)

<!-- ## Citation
```

``` -->
