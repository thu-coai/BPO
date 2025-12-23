import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from ranking_utils import (
    load_ranking_prompt,
    read_jsonl,
    run_ranking_loop,
    run_eval_model
)

device = 'cuda:0'
model_name = "unsloth/gemma-3-27b-it-bnb-4bit"

input_jsonl = "optimized_prompts_llama2_7b_res.jsonl"
output_jsonl = "lose_pairwise_results.jsonl"
bpo_jsonl = "optimized_prompts_llama2_7b_res-original.jsonl"

# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH)

# ==== LOAD PROMPT TEMPLATE ====
raw_prompt = load_ranking_prompt()

# ==== RUN ====
if __name__ == "__main__":
    rows = read_jsonl(input_jsonl)
    bpo_rows = read_jsonl(bpo_jsonl)

    # Merge rows với bpo_rows
    merged_rows = []
    for item, bpo_item in zip(rows, bpo_rows):
        merged = item.copy()
        merged["bpo_optimized_res"] = bpo_item["optimized_res"]
        merged["bpo_optimized_prompt"] = bpo_item["optimized_prompt"]
        merged_rows.append(merged)

    # Tạo eval function với model/tokenizer đã load
    eval_fn = lambda prompt: run_eval_model(prompt, model, tokenizer, device)

    run_ranking_loop(
        rows=merged_rows,
        eval_fn=eval_fn,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["bpo_prompt"],
        get_prompt_2_fn=lambda item: item["optimized_prompt"],
        get_output_1_fn=lambda item: item["bpo_res"],
        get_output_2_fn=lambda item: item["optimized_res"],
        label_0="original_win",
        label_1="bpo_win",
        label_2="draw",
        save_winner_0=True,
        save_winner_1=False,
    )
