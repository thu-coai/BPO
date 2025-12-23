from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
from config import MODEL_CACHE_PATH, prompt_template_vicuna
from utils import generate_batch

device = 'cuda:0'
model_name = "meta-llama/Llama-2-7b-chat-hf"
batch_size = 8  # Điều chỉnh tùy theo VRAM

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH, torch_dtype=torch.bfloat16, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH, legacy=False)

# Input & output JSONL
input_jsonl = "optimized_prompts.jsonl"
output_jsonl = "optimized_prompts_llama2_7b_res-original.jsonl"

# ---- READ JSONL ----
data = []
with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# ---- INFER BATCH ----
with torch.no_grad():
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for batch_start in tqdm(range(0, len(data), batch_size), desc="Inferring batches"):
            batch_data = data[batch_start:batch_start + batch_size]

            # Chuẩn bị batch prompts
            # prompts = [prompt_template_vicuna.format(item['optimized_prompt']) for item in batch_data]
            prompts = [item['optimized_prompt'] for item in batch_data]

            # Generate batch
            outputs = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=1.0,
                apply_chat_template=True,
                device=device
            )

            # Ghi từng kết quả
            for item, optimized_res in zip(batch_data, outputs):
                item['optimized_res'] = optimized_res
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Clear cache giữa các batch
            torch.cuda.empty_cache()

print("Done! Saved to:", output_jsonl)