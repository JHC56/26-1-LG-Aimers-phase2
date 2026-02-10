import os
import torch
import gc
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from google.colab import files

# 1. 설정 (안전한 타협안)
MODEL_ID = "/content/base_model"
OUT_DIR  = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# 샘플 수는 512로 타협(256보단 높음), 길이는 512로 고정하여 RAM 폭발 방지
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]

# 2. 모델 로드 (low_cpu_mem_usage는 필수)
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

# 3. 데이터셋 전처리
print("[INFO] 데이터 전처리 중...")
ds = load_dataset(DATASET_ID, split="train")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

ds = ds.map(preprocess, remove_columns=ds.column_names)

# 4. RAM 청소 (매우 중요)
gc.collect()
torch.cuda.empty_cache()

# 5. GPTQ 양자화
print(f"[INFO] GPTQ 시작 (메모리 절약 모드)...")
recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE,
        dampening_frac=0.1
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 6. 저장 및 압축 (이후 동일)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

zip_name = "submit06"
shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)
files.download(f"{zip_name}.zip")
print("[완료] 세션 종료 없이 작업이 끝났습니다.")
