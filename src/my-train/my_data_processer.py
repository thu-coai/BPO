# Template functions
import numpy as np
from transformers import AutoTokenizer

from config import CFG, MODEL_CACHE_PATH

MODEL_NAME: str = str(CFG["model"]["name"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def make_prompt_template(prompt, optimized_prompt=None, template_type="default"):
    if template_type == "default":
        prompt_text = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]".format(prompt)
    elif template_type == "llama":
        prompt_text = prompt
    elif template_type == "tiger":
        prompt_text = "\n\n### Instruction:\n" + prompt + "\n\n### Response:\n"
    else:
        prompt_text = prompt

    if optimized_prompt is not None:
        prompt_text += optimized_prompt

    return prompt_text

def format_example(example, template_type="default"):
    """
    example: dict {"prompt": ..., "optimized_prompt": ...}
    template_type: "default", "llama", "tiger"
    Lưu ý: Không padding ở đây, để collator xử lý batch
    """
    # --- Chọn template ---
    prompt_text = make_prompt_template(
        example["prompt"], optimized_prompt=None, template_type=template_type
    )

    # --- Encode ---
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(example["optimized_prompt"], add_special_tokens=False) + [tokenizer.eos_token_id]

    # --- Merge ---
    input_ids = prompt_ids + answer_ids

    # --- Labels ---
    # Mask prompt, loss chỉ tính trên answer
    labels = [-100] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,  # list of ids, padding để collator xử lý
        "labels": labels
    }