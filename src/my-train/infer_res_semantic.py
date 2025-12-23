import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from config import MODEL_CACHE_PATH
from utils import spherical_mean

# === CONFIG ===
device = 'cuda:0'

semantic_jsonl = "responses_with_semantic.jsonl"
output_jsonl = "optimized_prompts_llama2_7b_res.jsonl"

sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device, cache_folder=MODEL_CACHE_PATH)

# === LOAD DATA ===
semantic_data = [json.loads(l) for l in open(semantic_jsonl, 'r', encoding='utf-8')]

# def get_optimized_response(sem_item, threshold=0.2):
#     """
#     - Nếu conf_score < threshold -> trả về response_original
#     - Nếu conf_score >= threshold -> lấy đại diện trong nhóm có nhiều paraphrase_responses nhất

#     sem_item có các field:
#         - prompt
#         - response_original
#         - paraphrase_responses
#         - clusters: list of list (mỗi cluster chứa index của paraphrase_responses)
#         - cluster_representatives: list index đại diện mỗi cluster
#         - cluster_probs
#         - semantic_entropy
#         - conf_score
#     """
#     conf_score = sem_item.get("conf_score", 0)

#     # Nếu conf_score thấp -> trả về response gốc
#     if conf_score < threshold:
#         return sem_item.get("prompt", ""), sem_item.get("response_original", "")

#     # Lấy cluster lớn nhất
#     clusters = sem_item.get("clusters", [])
#     cluster_representatives = sem_item.get("cluster_representatives", [])
#     paraphrase_prompts = sem_item.get("paraphrase_prompts", [])
#     paraphrase_responses = sem_item.get("paraphrase_responses", [])

#     # Tìm cluster có nhiều phần tử nhất
#     largest_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))

#     # Lấy representative của cluster lớn nhất
#     best_rep_idx = cluster_representatives[largest_cluster_idx]

#     # print(sims)
#     # print(len(clusters[largest_cluster_idx]), end=' ')

#     return paraphrase_prompts[best_rep_idx], paraphrase_responses[best_rep_idx]

def get_optimized_response(sem_item, threshold=0.1):
    """
    - Nếu conf_score < threshold -> trả về response gốc
    - Nếu conf_score >= threshold -> chọn representative gần centroid nhất
    """
    # reps = sem_item.get("cluster_representatives", [])
    reps = [i for i in range(len(sem_item.get("paraphrase_responses", [])))]
    paraphrase_prompts = sem_item.get("paraphrase_prompts", [])
    paraphrase_responses = sem_item.get("paraphrase_responses", [])
    conf_score = sem_item.get("conf_score", 0)

    # Nếu conf_score thấp -> trả về response gốc
    if conf_score < threshold:
        return paraphrase_prompts[0], paraphrase_responses[0], sem_item.get("prompt", ""), sem_item.get("response_original", "")

    # Tạo tensor embeddings (n, d)
    rep_embs = sbert.encode([paraphrase_responses[i] for i in reps], convert_to_tensor=True)

    # Tính centroid (mean embedding)
    centroid = spherical_mean(rep_embs)  # shape (d)

    # Tính similarity giữa centroid và từng representative
    sims = util.pytorch_cos_sim(centroid, rep_embs)  # shape (n,)

    # Chọn representative gần centroid nhất
    best_idx = torch.argmax(sims).item()

    # print(sims)
    # print(reps[best_idx], end=' ')

    return paraphrase_prompts[0], paraphrase_responses[0], paraphrase_prompts[reps[best_idx]], paraphrase_responses[reps[best_idx]]

# === BẮT ĐẦU INFER ===
with torch.no_grad():
    with open(output_jsonl, 'w', encoding='utf-8') as fout:

        for item in tqdm(semantic_data, desc="Processing"):

            prompt = item["prompt"]

            out_item = {}
            out_item["prompt"] = prompt

            bpo_prompt, bpo_res, optimized_prompt, optimized_res = get_optimized_response(item)

            out_item["bpo_prompt"] = bpo_prompt
            out_item["bpo_res"] = bpo_res
            out_item["optimized_prompt"] = optimized_prompt
            out_item["optimized_res"] = optimized_res
            out_item["res"] = item["response_original"]

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

print("Done! Saved to:", output_jsonl)
