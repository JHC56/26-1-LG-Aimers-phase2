import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# [베이스라인 파일 설정 그대로 반영]
MODEL_ID = "/content/base_model"
OUT_DIR  = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE  = ["embed_tokens", "lm_head"]

# 1. 모델 로드 (KeyError: 'exaone4' 방지 버전)
print("[INFO] 모델 로드 중... (최신 라이브러리 적용)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True  # 이 옵션이 'exaone4' 아키텍처를 인식시킵니다.
)
print("[INFO] 모델/토크나이저 로드 완료")

# 2. 데이터셋 전처리
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}
ds = ds.map(preprocess)

# 3. GPTQ 양자화
print(f"[INFO] GPTQ 시작 (scheme={SCHEME})...recipe = [GPTQModifier(scheme=SCHEME, targets=TARGETS, ignore=IGNORE)]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 4. 저장
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)
print(f"✅ 모든 과정 완료! {OUT_DIR} 폴더를 확인하세요.")
