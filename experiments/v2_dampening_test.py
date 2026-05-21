
import os
import torch
import gc
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Configuration
MODEL_ID = "/content/base_model"
OUT_DIR = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]  # MLP 보호 없음

# Model loading
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

# Dataset (shuffle 적용)
ds = load_dataset(DATASET_ID, split="train")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

gc.collect()
torch.cuda.empty_cache()

# Quantization — dampening_frac=0.1 (실험)
recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        dampening_frac=0.1,  # ← 이것이 핵심 변수. 0.01 대비 큰 폭 하락.
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
