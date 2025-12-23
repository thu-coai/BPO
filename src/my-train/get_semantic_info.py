import json

input_path = "responses_with_semantic.jsonl"
output_path = "clusters_only.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        
        obj = json.loads(line)
        # Lấy các trường cần thiết
        clusters = obj.get("clusters")
        cluster_probs = obj.get("cluster_probs")
        semantic_entropy = obj.get("semantic_entropy")
        conf_score = obj.get("conf_score")
        
        # Chỉ ghi nếu tồn tại clusters
        if clusters is not None:
            fout.write(json.dumps({
                "clusters": clusters,
                "cluster_probs": cluster_probs,
                "semantic_entropy": semantic_entropy,
                "conf_score": conf_score
            }, ensure_ascii=False) + "\n")

print("Done! File mới đã được tạo:", output_path)
