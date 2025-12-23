import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from config import CFG, DATA_CACHE_PATH, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH
from my_data_processer import make_prompt_template


MODEL_NAME: str = str(CFG["model"]["name"])
TRAIN_PATH: str = str(CFG["dataset"]["train_path"])
VAL_PATH: str = str(CFG["dataset"].get("val_path", None))
MAX_TOKEN: int = int(CFG["dataset"]["max_token"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
# Ensure left padding for Flash Attention / Qwen3
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def add_dataset(
    all_data,
    path,
    prompt_key,
    optimized_prompt_key,
    *,
    subset=None,
    split="train",
    n_samples=None,
    max_token=768,
):
    
    print(f"📥 Loading {path}...")
    ds = load_dataset(path, subset, split=split, cache_dir=DATA_CACHE_PATH)
    ds = ds.shuffle(seed=42)

    if n_samples is None:
        n_samples = len(ds)

    candidates = []
    for ex in ds:
        if len(candidates) >= n_samples:
            break

        prompt = ex[prompt_key]
        optimized_prompt = ex[optimized_prompt_key]

        prompt_text = make_prompt_template(prompt, optimized_prompt=optimized_prompt, template_type="default")

        tokens = tokenizer(prompt_text, truncation=False, add_special_tokens=False)
        tok_len = len(tokens["input_ids"])
        if tok_len > max_token:
            continue

        candidates.append({
            "prompt": prompt.strip(),
            "optimized_prompt": optimized_prompt.strip(),
        })

    all_data.extend(candidates)

def export_dataset(all_data, path):
    os.makedirs(INSTRUCTION_DATA_PATH, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n🏁 Done! Total: {len(all_data)} samples")

if __name__ == "__main__":
    train_data = []
    add_dataset(
        train_data,
        path="zai-org/BPO",
        prompt_key="prompt",
        split='train',
        optimized_prompt_key="optimized_prompt",
        max_token=MAX_TOKEN,
    )
    export_dataset(train_data, path=TRAIN_PATH)

    if VAL_PATH is not None:
        val_data = []
        add_dataset(
            val_data,
            path="zai-org/BPO",
            prompt_key="prompt",
            split='validation',
            optimized_prompt_key="optimized_prompt",
            max_token=MAX_TOKEN,
        )
        export_dataset(val_data, path=VAL_PATH)



