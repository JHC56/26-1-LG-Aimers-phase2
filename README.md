# EXAONE 4.0 1.2B 경량화 (GPTQ)

LG AIMers 8기 Phase 2 온라인 해커톤 (DACON) 참가 기록.  
EXAONE 4.0 1.2B 모델을 W4A16 GPTQ로 경량화했고, 최종 Public Score **0.6184**를 기록했다.

팀: CJH4567(최주혁), gmk(강민기)

---

## 구조

```
experiments/
  v1_baseline_059.py      첫 시도 (0.59)
  v2_dampening_test.py    dampening_frac 실험 (0.47)
  v3_best_061.py          최종 제출 코드 (0.6184)
  actorder_variant.py     actorder 실험 (실패)
docs/
  experiment_log.py       전체 실험 기록 (실행 가능)
baseline/
  dacon_baseline.py       DACON 제공 베이스라인 (~0.50)
utils/
  compress.py             제출용 zip 생성
```

---

## 왜 MLP를 보호해야 하는가

EXAONE 1.2B의 파라미터 분포를 보면 MLP가 70.6%, Attention이 29.4%다.  
MLP를 양자화하면 성능이 바로 무너진다. 실제로 MLP 보호 없이 전체 양자화하면 0.47까지 떨어졌다.

MLP를 통째로 보호하고 Attention만 4비트로 양자화하는 게 이 모델에서는 정답이었다.  
이걸 알아내기까지가 0.59에서 0.61로 가는 전환점이었다.

---

## 점수 추이

| 버전 | 설명 | Score | 시간 | 코드 |
|------|------|------:|-----:|------|
| baseline | DACON 베이스라인 | ~0.50 | - | `baseline/dacon_baseline.py` |
| v1 | 첫 GPTQ 적용, 256 samples | 0.5900 | 10m 14s | `experiments/v1_baseline_059.py` |
| v2 | MLP 보호 시작, 512 samples | 0.5991 | 10m 44s | - |
| v3 | MLP 보호 + 최적 설정 | 0.6136 | 10m 7s | `experiments/v3_best_061.py` |
| v4 | v3 재제출 (서버 상태 차이) | **0.6184** | 10m 2s | 동일 |

v3과 v4는 같은 코드다. 서버 상태에 따라 0.613~0.618 사이에서 변동이 있었다.

---

## 실패한 것들

| 실험 | Score | 왜 실패했는가 |
|------|------:|-------------|
| dampening_frac 0.1 | 0.4699 | 감쇠값이 너무 커서 양자화 정밀도 붕괴 |
| group size 64 | 0.5806 | 정밀도는 올라가는데 속도가 더 떨어짐 |
| group size 32 | 0.5473 | 더 악화. actorder는 라이브러리가 지원 안 함 |
| MLP 보호 없이 전체 양자화 | 0.4715 | MLP를 건드리면 안 된다는 걸 증명 |
| MLP 부분 보호 (gate/up/down 개별) | 0.5240 | 개별 보호는 전체 보호의 절반도 안 됨 |
| 레이어 20번부터 보호 | 0.5500 | 보호 레이어 늘리면 용량 커지고 속도 하락 |
| LoRA 파인튜닝 | 0.30~0.48 | 10스텝만 해도 기존 지식이 파괴됨 |
| 지식 증류 | ~0.30 | 같은 이유 |
| Pruning | 측정 불가 | 1.2B에서 효과 없음 |
| KMMLU 데이터셋 | 하락 | MANTA-1M이 이 모델에 가장 맞음 |
| 128 samples + 1024 len | 0.5110 | 샘플 수를 줄이면 안 됨 |
| 1024 samples | OOM | Colab 무료 메모리 한계 |

---

## 최적 설정

```python
GPTQModifier(
    scheme="W4A16",
    targets=["Linear"],
    ignore=["embed_tokens", "lm_head", "re:model\\.layers\\.\\d+\\.mlp\\..*"],
    dampening_frac=0.01,
    block_size=128,
)
# calibration: 512 samples, 512 seq_length
# dataset: LGAI-EXAONE/MANTA-1M
# seed: 42
```

---

## 배운 것

- 1.2B 모델은 설정값 하나에 0.1점이 왔다갔다 한다. 큰 모델과는 다르다.
- 이 대회에서는 지능보다 속도가 점수에 더 크게 반영됐다.
- 파인튜닝(LoRA, 증류)은 소형 모델에서 역효과가 난다. 기존 지식이 너무 쉽게 깨진다.
- 서버 상태에 따라 같은 코드도 0.005~0.012점 차이가 난다 (DACON Q&A 공식 확인).
- 결국 0.62 넘기는 건 이 설정에서는 운의 영역이었다.

---

## 환경

```
pip install llmcompressor peft datasets==4.4.1 accelerate==1.10.1
pip install transformers==4.57.3  # 대회 서버 호환 버전. 꼭 고정해야 함.
```

Google Colab T4 GPU에서 작업했다.  
base_model은 대회 제공 모델이라 이 레포에 포함되어 있지 않다.  
HuggingFace에서 `LGAI-EXAONE/EXAONE-4.0-1.2B`를 받으면 된다.

---

## 참고

- LG AI Research. (2024). EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes. arXiv:2412.06450
- [LGAI-EXAONE/MANTA-1M](https://huggingface.co/datasets/LGAI-EXAONE/MANTA-1M)
- [llmcompressor](https://github.com/vllm-project/llmcompressor)
