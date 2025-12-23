import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from utils import generate_batch, spherical_mean

from config import (
    MODEL_CACHE_PATH,
    prompt_template_optimize,
    prompt_template_vicuna
)

# ============ CONFIG ============
device = "cuda:0"
input_jsonl = "optimized_prompts.jsonl"
tmp_step1 = "tmp_step1_r0.jsonl"
tmp_step2 = "tmp_step2_r0.jsonl"
output_jsonl = "responses_with_semantic.jsonl"
infer_model_path = "meta-llama/Llama-2-7b-chat-hf"  # chỉnh nếu cần
M = 10

# -----------------------------------------------------
# Helper: generate text
# -----------------------------------------------------


# -----------------------------------------------------
# STEP 1: Generate paraphrase prompts using BPO 
# -----------------------------------------------------
def step1_generate_paraphrase():
    print("\n===== STEP 1: Paraphrase =====")

    optimize_path = "THUDM/BPO"
    model = AutoModelForCausalLM.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, dtype=torch.float16
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, use_fast=False, legacy=True
    )
    model.config.return_dict = True

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(tmp_step1, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step1"):
            item = json.loads(line)
            prompt = item["prompt"]

            # Sinh M paraphrases (batch)
            batch_prompts = [prompt_template_optimize.format(prompt) for _ in range(M)]
            paraphrases = generate_batch(
                model, tokenizer,
                batch_prompts,
                temperature=1.0,
                top_p=0.9,
                apply_chat_template=False,
                device=device
            )

            fout.write(json.dumps({
                "prompt": prompt,
                "paraphrase_prompts": paraphrases
            }, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    print("✓ Done Step 1 →", tmp_step1)

# -----------------------------------------------------
# STEP 2: LLM inference cho từng paraphrase (Vicuna)
# Input  : tmp_step1  (từ Step 1, chứa prompt + paraphrase_prompts)
# Output : tmp_step2  (mỗi item sẽ có thêm paraphrase_responses)
# -----------------------------------------------------
# -----------------------------------------------------
# STEP 2: LLM inference cho từng paraphrase (dùng hàm generate() có sẵn)
# -----------------------------------------------------
def step2_infer_vicuna(device="cuda:0"):
    print("===== STEP 2: Vicuna inference (full, paraphrases + original prompt) =====")

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        infer_model_path,
        cache_dir=MODEL_CACHE_PATH,
        torch_dtype=torch.float16
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        infer_model_path,
        cache_dir=MODEL_CACHE_PATH,
        legacy=False
    )

    batch_size = 6  # Điều chỉnh tùy theo VRAM

    # File input/output
    with open(tmp_step1, "r", encoding="utf-8") as fin, \
         open(tmp_step2, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step 2 - infer prompts"):
            item = json.loads(line)
            prompt = item["prompt"]
            paraphrases = item["paraphrase_prompts"]

            # === 1) Lọc các prompt unique (bao gồm cả original) ===
            all_prompts = [prompt.strip()] + [p.strip() for p in paraphrases]

            # Tạo mapping: unique_prompt -> index đầu tiên xuất hiện
            unique_prompts = []
            prompt_to_idx = {}  # prompt -> index trong unique_prompts
            original_to_unique = []  # index trong all_prompts -> index trong unique_prompts

            for p in all_prompts:
                if p not in prompt_to_idx:
                    prompt_to_idx[p] = len(unique_prompts)
                    unique_prompts.append(p)
                original_to_unique.append(prompt_to_idx[p])

            # === 2) Infer chỉ các prompt unique ===
            unique_responses = []
            for batch_start in range(0, len(unique_prompts), batch_size):
                batch = unique_prompts[batch_start:batch_start + batch_size]

                batch_responses = generate_batch(
                    model,
                    tokenizer,
                    batch,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=1.0,
                    apply_chat_template=True,
                    device=device
                )
                unique_responses.extend(batch_responses)
                torch.cuda.empty_cache()

            # === 3) Map lại response cho tất cả prompts (bao gồm trùng) ===
            all_responses = [unique_responses[original_to_unique[i]] for i in range(len(all_prompts))]

            # Tách response_original và paraphrase_responses
            response_original = all_responses[0]
            paraphrase_responses = all_responses[1:]

            # === Lưu output STEP 2 ===
            fout.write(json.dumps({
                "prompt": prompt,
                "response_original": response_original,
                "paraphrase_prompts": paraphrases,
                "paraphrase_responses": paraphrase_responses
            }, ensure_ascii=False) + "\n")

    # cleanup
    del model
    torch.cuda.empty_cache()
    print("✓ Done STEP 2 →", tmp_step2)

# -----------------------------------------------------
# STEP 3: SBERT clustering + semantic entropy 
# -----------------------------------------------------
def step3_sbert_clustering(device='cuda:0', threshold=0.9):
    print("===== STEP 3: SBERT clustering + semantic entropy =====")

    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device, cache_folder=MODEL_CACHE_PATH)

    with open(tmp_step2, "r", encoding="utf-8") as fin, \
        open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            item = json.loads(line)
            samples = item["paraphrase_prompts"]

            # Encode tất cả các câu
            embeddings = sbert.encode(samples, convert_to_tensor=True)

            # Khởi tạo cluster
            clusters = []
            used = set()

            for i, emb_i in enumerate(embeddings):
                if i in used:
                    continue
                cluster = [i]
                used.add(i)
                for j, emb_j in enumerate(embeddings):
                    if j in used:
                        continue
                    sim = util.pytorch_cos_sim(emb_i, emb_j).item()
                    if sim >= threshold:
                        cluster.append(j)
                        used.add(j)
                clusters.append(cluster)

            # ====== 🔥 Chọn đại diện cho mỗi cluster ======
            cluster_representatives = []

            for cluster in clusters:
                # Nếu cụm có 1 phần tử → chính nó là đại diện
                if len(cluster) == 1:
                    cluster_representatives.append(cluster[0])
                    continue

                # Lấy embedding của cluster
                cluster_embeds = torch.stack([embeddings[i] for i in cluster])

                # Tính centroid (mean vector)
                centroid = spherical_mean(cluster_embeds)

                # Tính cosine similarity giữa centroid và từng embedding trong cluster
                sims = util.pytorch_cos_sim(centroid, cluster_embeds)

                # Chọn index trong cluster gần centroid nhất (cosine sim cao nhất)
                best_local_idx = torch.argmax(sims).item()

                # Map lại index này về index trong samples
                best_idx = cluster[best_local_idx]

                # Thêm representative
                cluster_representatives.append(best_idx)

            # Cluster probabilities
            cluster_probs = [len(c)/len(samples) for c in clusters]

            # Semantic entropy
            entropy = -sum(p * (math.log(p) if p > 0 else 0) for p in cluster_probs)

            # Confidence score
            K = len(clusters)
            conf_score = 1 - (entropy / math.log(K)) if K > 1 else 1.0

            # ==== Ghi ra JSONL có thêm cluster_representatives ====
            fout.write(json.dumps({
                "prompt": item["prompt"],
                "response_original": item["response_original"],
                "paraphrase_responses": item["paraphrase_responses"],
                "paraphrase_prompts": item["paraphrase_prompts"],
                "clusters": clusters,
                "cluster_representatives": cluster_representatives,
                "cluster_probs": cluster_probs,
                "semantic_entropy": entropy,
                "conf_score": conf_score
            }, ensure_ascii=False) + "\n")

        print("✓ Done STEP 3 (SBERT)")

# -----------------------------------------------------
# RUN ALL STEPS
# -----------------------------------------------------
if __name__ == "__main__":
    # step1_generate_paraphrase()
    # step2_infer_vicuna()
    step3_sbert_clustering()
    print("\n🎉 ALL DONE!")
