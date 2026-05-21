# LG Aimers 8기 — EXAONE 4.0 1.2B 경량화

## 대회 개요

| | |
|---|---|
| 주제 | EXAONE 4.0 1.2B 모델 경량화 (GPTQ Quantization) |
| 기간 | 2026.01.02 ~ 2026.02.26 |
| 주최 | LG AI Research × DACON |

---

## 실험 기록

| 실험 | 코드 | Score | 시간 |
|------|------|------:|-----:|
| DACON 베이스라인 | [dacon_baseline.py](baseline/dacon_baseline.py) | ~0.50 | - |
| 첫 GPTQ 적용 (256 samples) | [v1_baseline_059.py](experiments/v1_baseline_059.py) | 0.5900 | 10m 14s |
| dampening_frac 실험 | [v2_dampening_test.py](experiments/v2_dampening_test.py) | 0.4699 | 13m 59s |
| MLP 보호 + 샘플 512 | - | 0.5991 | 10m 44s |
| MLP 보호 최적화 | [v3_best_061.py](experiments/v3_best_061.py) | 0.6136 | 10m 7s |
| actorder 실험 | [actorder_variant.py](experiments/actorder_variant.py) | 미지원 | - |
| group size 64 | - | 0.5806 | 11m 11s |
| group size 32 | - | 0.5473 | 11m 43s |
| MLP 보호 없이 전체 양자화 | - | 0.4715 | 13m 15s |
| MLP 부분 보호 (gate/up/down 개별) | - | 0.5240 | 11m+ |
| 레이어 20번부터 보호 | - | 0.5500 | - |
| LoRA 파인튜닝 (10~32 step) | - | 0.30~0.48 | - |
| 지식 증류 | - | ~0.30 | - |
| KMMLU 데이터셋 | - | 하락 | - |

전체 실험 기록은 [experiment_log.py](docs/experiment_log.py)에서 실행해서 볼 수 있다.

---

## 최종 전략

EXAONE 1.2B는 MLP가 파라미터의 70.6%를 차지한다.  
MLP를 양자화하면 성능이 바로 무너지고, MLP를 보호한 채 Attention(29.4%)만 4비트 양자화하는 게 최적이었다.

```python
GPTQModifier(
    scheme="W4A16",
    targets=["Linear"],
    ignore=["embed_tokens", "lm_head", "re:model\\.layers\\.\\d+\\.mlp\\..*"],
    dampening_frac=0.01,
    block_size=128,
)
# 512 samples, 512 seq_length, seed=42
# dataset: LGAI-EXAONE/MANTA-1M
```

구현 코드: [v3_best_061.py](experiments/v3_best_061.py)

---

## 대회 결과

| | Score |
|---|------:|
| Public (Best) | **0.6184** |

---

## 환경

```
transformers==4.57.3 (대회 서버 호환 버전)
llmcompressor, datasets==4.4.1, accelerate==1.10.1
Google Colab T4 GPU
```

---

## 참고

- LG AI Research. (2024). EXAONE 4.0. arXiv:2412.06450
- [LGAI-EXAONE/MANTA-1M](https://huggingface.co/datasets/LGAI-EXAONE/MANTA-1M)
