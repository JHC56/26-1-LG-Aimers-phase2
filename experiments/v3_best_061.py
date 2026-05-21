
import os
import torch
import gc
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "/content/base_model"
OUT_DIR = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 512

SCHEME = "W4A16"
TARGETS = ["Linear"]

IGNORE_STRATEGY = [
    "embed_tokens",
    "lm_head",
    "re:model\\.layers\\.\\d+\\.mlp\\..*",
]


print("[1/4] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("[2/4] 캘리브레이션 데이터 준비 중...")
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

print("[3/4] GPTQ 양자화 시작 (MLP 보호 적용)...")
recipe = [
    GPTQModifier(
        scheme=SCHEME,
        targets=TARGETS,
        ignore=IGNORE_STRATEGY,
        dampening_frac=0.01,  # 0.1은 0.47로 추락. 0.01이 최적.
        block_size=128,       # 64→0.58, 32→0.55. 128이 최적.
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("[4/4] 모델 저장 중...")
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# 제출용 zip
zip_name = "submit_final"
shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)

print(f"[완료] 양자화 모델: {OUT_DIR}")
print(f"[완료] 제출 파일: {zip_name}.zip")