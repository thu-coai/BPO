import torch
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from ranking_utils import (
    load_ranking_prompt,
    read_jsonl,
    run_ranking_loop,
    run_followup_inference
)

device = 'cuda:0'
model_name = "meta-llama/Llama-2-7b-chat-hf"

input_jsonl = "optimized_prompts_llama2_7b_res.jsonl"
output_jsonl = "lose_pairwise_results.jsonl"

# ==== LOAD MODEL (for followup inference) ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH)

# ==== GEMINI CLIENT ====
API_KEY = "api-key"
client = genai.Client(api_key=API_KEY)

# ==== LOAD PROMPT TEMPLATE ====
raw_prompt = load_ranking_prompt()

# ==== EVAL FUNCTION (Gemini + local model) ====
def run_with_gemini(prompt):
    """
    Gọi Gemini để lấy reasoning, sau đó dùng local model để extract boxed answer
    """
    # 1. Gọi Gemini để xác định model rank
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    reasoning = response.text.strip()

    # 2. Dùng hàm chung để extract boxed answer
    return run_followup_inference(reasoning, model, tokenizer, device)

# ==== RUN ====
if __name__ == "__main__":
    rows = read_jsonl(input_jsonl)

    run_ranking_loop(
        rows=rows,
        eval_fn=run_with_gemini,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_optimized_instruction_fn=lambda item: item["optimized_prompt"],
        get_output_1_fn=lambda item: item["optimized_res"],
        get_output_2_fn=lambda item: item["res"],
        label_0="optimized_win",
        label_1="original_win",
        label_2="draw",
        save_winner_0=False,
        save_winner_1=True,
    )
