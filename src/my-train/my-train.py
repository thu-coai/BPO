import random
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from peft import LoraConfig, get_peft_model, TaskType
from config import CFG, INSTRUCTION_DATA_PATH, MODEL_CACHE_PATH
from datasets import load_dataset
from my_data_processer import format_example, tokenizer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
random.seed(42)

# --- Cấu hình ---
# Ví dụ: lấy các giá trị
MODEL_NAME: str = str(CFG["model"]["name"])

OUTPUT_DIR: str = str(CFG["training"]["output_dir"])
LR: float = float(CFG["training"]["learning_rate"])
BATCH_SIZE: int = int(CFG["training"]["batch_size"])
EPOCHS: int = int(CFG["training"]["epochs"])

LORA_CONFIG: dict = dict(CFG["lora"])
TRAIN_PATH: str = str(CFG["dataset"]["train_path"])
VAL_PATH: str = str(CFG["dataset"].get("val_path", None))
VAL_RATIO: float = float(CFG["dataset"].get("val_ratio", None))

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # tiết kiệm VRAM
    device_map='cuda',
    attn_implementation="flash_attention_2",
    cache_dir=MODEL_CACHE_PATH,
)

# --- Thiết lập LoRA ---
if bool(LORA_CONFIG["using_lora"]):
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

if __name__ == "__main__":
    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=2*BATCH_SIZE,
        learning_rate=LR,
        weight_decay=CFG["training"]["weight_decay"],
        bf16=True,
        tf32=True,
        gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
        logging_steps=CFG["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=CFG["training"]["eval_steps"],
        save_steps=CFG["training"]["save_steps"],
        save_total_limit=CFG["training"]["save_total_limit"],
        report_to="tensorboard",
        logging_dir="./logs",
        lr_scheduler_type=CFG["training"]["lr_scheduler_type"],
        warmup_ratio=CFG["training"]["warmup_ratio"],
        load_best_model_at_end=True,
        label_names=["labels"]
    )

    # --- Load dataset ---
    train_dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH+"train.json")["train"]
    if VAL_PATH is not None:
        val_dataset = load_dataset("json", data_files=INSTRUCTION_DATA_PATH+"val.json")["train"]
    else:
        dataset_split = dataset.train_test_split(test_size=VAL_RATIO, seed=42)
        train_dataset = dataset_split["train"]
        val_dataset = dataset_split["test"]

    train_tokenized_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    val_tokenized_dataset = val_dataset.map(format_example, remove_columns=val_dataset.column_names)

    # Thêm Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_TOKEN_ID, # Rất quan trọng!
        padding="longest",
        pad_to_multiple_of=8 # Tùy chọn, giúp tối ưu hóa trên GPU tensor core
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,   # ✅ thêm tập validation
        data_collator=data_collator,
    )

    # --- Train ---
    trainer.train()
    trainer.evaluate()

    # --- Lưu model ---
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Huấn luyện hoàn tất, model lưu tại:", OUTPUT_DIR)