def make_prompt_template(user_prompt: str, add_system_prompt=True):
    messages = []
    if add_system_prompt:
        messages.append({
            "role": "system",
            "content": "You are a helpful and concise assistant. "
                    "Please reply in English only."
        })
    messages.append({
        "role": "user",
        "content": user_prompt
    })

    return messages


def generate(model, tokenizer, prompt, max_new_tokens=1024, apply_chat_template=True, do_sample=True, device="cuda", **kwargs):
    """Generate cho single prompt (backward compatible)"""
    if apply_chat_template:
        prompt = make_prompt_template(prompt)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    # Generate
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        **kwargs
    )

    # Lấy phần sinh thêm
    generated_ids = out[0][len(input_ids[0]):]

    # Decode chỉ phần mới
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text.strip()


def generate_batch(model, tokenizer, prompts, max_new_tokens=1024, apply_chat_template=True, do_sample=True, device="cuda", **kwargs):
    """Generate cho batch prompts - tăng tốc inference"""
    if not prompts:
        return []
    tokenizer.padding_side='left'

    # Apply chat template nếu cần
    if apply_chat_template:
        processed_prompts = []
        for p in prompts:
            p = make_prompt_template(p)
            p = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            processed_prompts.append(p)
        prompts = processed_prompts

    # Set pad token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize batch với padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Lưu độ dài input của từng prompt (trước padding)
    input_lengths = [len(ids) for ids in inputs["input_ids"]]

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs
    )

    # Decode từng output, cắt bỏ phần input
    results = []
    for i, output in enumerate(outputs):
        # Cắt bỏ phần input prompt
        generated_ids = output[input_lengths[i]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text.strip())

    return results

import torch
import torch.nn.functional as F

def spherical_mean(embeddings: torch.Tensor, eps: float = 1e-8):
    """
    embeddings: Tensor (n, d), CHƯA hoặc ĐÃ normalize đều được
    return: centroid (d,), L2-normalized

    Dùng cho cosine similarity / spherical k-means
    """
    # Normalize từng vector
    X = F.normalize(embeddings, dim=1, eps=eps)

    # Tổng vector
    s = X.sum(dim=0)

    # Normalize lại centroid
    centroid = F.normalize(s, dim=0, eps=eps)
    return centroid
