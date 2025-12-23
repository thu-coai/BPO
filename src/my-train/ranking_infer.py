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

    # Tạo eval function với model/tokenizer đã load
    eval_fn = lambda prompt: run_eval_model(prompt, model, tokenizer, device)

    run_ranking_loop(
        rows=rows,
        eval_fn=eval_fn,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_optimized_instruction_fn=lambda item: item["optimized_prompt"],
        get_output_1_fn=lambda item: item["res"],
        get_output_2_fn=lambda item: item["optimized_res"],
        label_0="original_win",
        label_1="optimized_win",
        label_2="draw",
        save_winner_0=True,
        save_winner_1=False,
    )
