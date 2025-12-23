# ⚙️ Cấu hình
import torch
import yaml

with open("config.yml", "r") as f:
    CFG = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_PATH = "./hf_cache/model"
DATA_CACHE_PATH = "./hf_cache/data"
INSTRUCTION_DATA_PATH = "data/"       # file jsonl như mô tả ở trên

prompt_template_optimize = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"
prompt_template_vicuna = """USER: You are a helpful and concise assistant. Please reply in English only.
{} 
ASSISTANT: """ 