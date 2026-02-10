import os
import torch
import gc
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from google.colab import files

# 1. 설정
MODEL_ID = "/content/base_model"
OUT_DIR  = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

SCHEME = "W4A16"
TARGETS = ["Linear"]

# model.layers.X.mlp 하위의 모든 레이어를 강제로 보호합니다.
IGNORE_STRATEGY = [
    "embed_tokens",
    "lm_head",
    "re:model\.layers\.\d+\.mlp\..*"
]

# 2. 모델 로드
print("[INFO] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 3. 데이터셋 전처리
print("[INFO] 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split="train")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=True,
        tokenize=False
    )}

ds = ds.map(preprocess, remove_columns=ds.column_names)

gc.collect()
torch.cuda.empty_cache()

# 4. GPTQ 양자화 (정규표현식 보호 적용)
print(f"[INFO] GPTQ 시작 (1.1GB 방지 강제 ignore 설정)...")

recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE_STRATEGY,
        dampening_frac=0.01,
        block_size=128
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 5. 저장 및 압축
print("[INFO] 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

zip_name = "submit_final_v6"
shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)
files.download(f"{zip_name}.zip")

print(f"[완료] 레이어 보호가 강제 적용되었습니다. 용량이 1.1GB보다 커야 정상입니다.")
