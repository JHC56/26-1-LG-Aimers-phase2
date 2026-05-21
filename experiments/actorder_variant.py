import os
import torch
import gc
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "/content/base_model"
OUT_DIR = "./model_actorder"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

ds = load_dataset(DATASET_ID, split="train", trust_remote_code=True)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    try:
        text = tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        text = str(example["conversations"])
    return {"text": text}

ds = ds.map(preprocess)

# actorder="static" — activation order 최적화 시도
recipe = [
    GPTQModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
        actorder="static",
        dampening_frac=0.01,
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
