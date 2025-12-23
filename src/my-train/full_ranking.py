import re
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CACHE_PATH
from utils import make_prompt_template

device = "cuda:0"
model_name = "unsloth/gemma-3-27b-it-bnb-4bit"

# ==== MAIN RANKING ====

input_file = "responses_with_semantic.jsonl"
output_file = "all_ranking_results.jsonl"

# ==== LOAD RANKING PROMPT TEMPLATE ====
with open("ranking_prompt.txt", "r", encoding="utf-8") as f:
    RAW_PROMPT = f.read()

def fill_prompt(instruction, output_1, output_2):
    """Thay vào template."""
    p = RAW_PROMPT
    p = p.replace('""{instruction}""', instruction)
    p = p.replace('""{output_1}""', output_1)
    p = p.replace('""{output_2}""', output_2)
    return p

# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH,
    dtype=torch.float32,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH)

def run_eval(prompt):
    # 1. Infer lần 1
    prompt = make_prompt_template(prompt, add_system_prompt=False)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    input_len = model_inputs["input_ids"].shape[1]
    output = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    # 3. Tách phần mới sinh (bỏ prompt)
    generated_ids = output[0][input_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
    print(tokenizer.decode(output[0], skip_special_tokens=False))

    # 3. Nối câu bắt buộc boxed để model infer tiếp
    followup_prompt = decoded + "\nSo among 'model_1', 'model_2' and 'both', we should choose \\boxed{"

    # 4. Infer tiếp
    model_inputs2 = tokenizer(followup_prompt, return_tensors="pt", truncation=True, max_length=5000).to(device)
    output2 = model.generate(
        **model_inputs2,
        max_new_tokens=50,  # đủ để điền tên model
        do_sample=False
    )
    decoded2 = tokenizer.decode(output2[0], skip_special_tokens=True)

    # 5. Lấy phần model sinh thêm và đóng dấu }
    final_output = followup_prompt + decoded2[len(followup_prompt):].strip()

    # print(final_output)

    return final_output

def extract_winner(text):
    """
    Tìm \boxed{…} đầu tiên trong text, xem có số 1 hay 2 bên trong.
    Trả về:
        0 nếu có số 1
        1 nếu có số 2
        None nếu không tìm thấy
    """
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        content = m.group(1)
        if 'both' in content:
            return 2
        if '1' in content:
            return 0
        elif '2' in content:
            return 1
    return None

# MẢNG KẾT QUẢ DUY NHẤT
winners = []

with open(input_file, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

with open(output_file, "w", encoding="utf-8") as fout:
    with open(input_file, "r", encoding="utf-8") as fin:

        for line in tqdm(fin, desc="Ranking all paraphrases"):
            item = json.loads(line)

            prompt = item["prompt"]
            original = item["response_original"]
            samples = item["samples"]

            winners = []

            for para in samples:

                if para.strip() == original.strip():
                    winners.append(2)
                    continue

                p = fill_prompt(prompt, para, original)
                rank_out = run_eval(p)
                w = extract_winner(rank_out)

                winners.append(w)

            # === CHỈ GHI MẢNG WINNERS ===
            fout.write(json.dumps(winners, ensure_ascii=False) + "\n")
            fout.flush()

print("DONE →", output_file)
