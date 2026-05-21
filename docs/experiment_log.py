"""
EXAONE 4.0 1.2B 경량화 — 전체 실험 기록

이 파일을 실행하면 모든 실험 결과를 한눈에 볼 수 있습니다.

    python docs/experiment_log.py

Author: team.CJH4567
"""


# 모델 구조 분석

MODEL_INFO = {
    "name": "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "total_params": "1.2B",
    "mlp_ratio": 0.706,
    "attention_ratio": 0.294,
    "num_layers": 24,
    "critical_layer": 20,  # 이 레이어부터 양자화 에러 급등
}


# 실험 결과


EXPERIMENTS = [
    # --- 성공한 버전들 (점수 상승 순) ---
    {
        "ver": "baseline",
        "desc": "DACON 제공 베이스라인",
        "score": 0.50,
        "time": "-",
        "config": "기본 GPTQ, 256 samples",
        "file": "baseline/dacon_baseline.py",
    },
    {
        "ver": "v1",
        "desc": "첫 GPTQ 적용",
        "score": 0.5900,
        "time": "10m 14s",
        "config": "W4A16, 256 samples, no MLP protect",
        "file": "experiments/v1_baseline_059.py",
    },
    {
        "ver": "v2",
        "desc": "MLP 보호 + 샘플 증가",
        "score": 0.5991,
        "time": "10m 44s",
        "config": "W4A16, 512 samples, MLP protect",
        "file": "-",
    },
    {
        "ver": "v3",
        "desc": "MLP 보호 최적화",
        "score": 0.6136,
        "time": "10m 7s",
        "config": "W4A16, 512 samples, MLP regex, dampening=0.01",
        "file": "experiments/v3_best_061.py",
    },
    {
        "ver": "v4 ★",
        "desc": "최고 점수 (v3 재제출)",
        "score": 0.6184,
        "time": "10m 2s",
        "config": "v3과 동일 (서버 상태 차이)",
        "file": "experiments/v3_best_061.py",
    },

    # --- 실패한 실험들 ---
    {
        "ver": "exp-damp",
        "desc": "dampening_frac 0.1",
        "score": 0.4699,
        "time": "13m 59s",
        "config": "dampening 10배 증가",
        "file": "experiments/v2_dampening_test.py",
    },
    {
        "ver": "exp-g64",
        "desc": "group size 64",
        "score": 0.5806,
        "time": "11m 11s",
        "config": "block_size 128→64",
        "file": "-",
    },
    {
        "ver": "exp-g32",
        "desc": "group size 32",
        "score": 0.5473,
        "time": "11m 43s",
        "config": "block_size 128→32, actorder 미지원",
        "file": "-",
    },
    {
        "ver": "exp-attn",
        "desc": "MLP 양자화 (보호 없음)",
        "score": 0.4715,
        "time": "13m 15s",
        "config": "MLP 보호 제거",
        "file": "-",
    },
    {
        "ver": "exp-mlp-p",
        "desc": "MLP 부분 보호",
        "score": 0.5240,
        "time": "11m+",
        "config": "gate/up/down 개별 보호",
        "file": "-",
    },
    {
        "ver": "exp-l20",
        "desc": "레이어 20+ 보호",
        "score": 0.5500,
        "time": "-",
        "config": "에러 급등 구간만 보호",
        "file": "-",
    },
    {
        "ver": "exp-lora",
        "desc": "LoRA 파인튜닝",
        "score": 0.30,
        "time": "-",
        "config": "10~32 steps → 지식 파괴",
        "file": "-",
    },
    {
        "ver": "exp-dist",
        "desc": "지식 증류",
        "score": 0.30,
        "time": "-",
        "config": "증류 → 동일 실패",
        "file": "-",
    },
    {
        "ver": "exp-kmmlu",
        "desc": "KMMLU 데이터셋",
        "score": None,
        "time": "-",
        "config": "외부 데이터셋 → 하락",
        "file": "-",
    },
    {
        "ver": "exp-s128",
        "desc": "128 samples + 1024 len",
        "score": 0.5110,
        "time": "-",
        "config": "샘플 줄이고 길이 늘림",
        "file": "-",
    },
    {
        "ver": "exp-act",
        "desc": "actorder static",
        "score": None,
        "time": "-",
        "config": "라이브러리 미지원",
        "file": "experiments/actorder_variant.py",
    },
]


# 체크리스트

CHECKLIST = [
    ("GPTQ 양자화",                    "0.59 → 0.618"),
    ("LoRA 파인튜닝",                   "실패 (0.30~0.48)"),
    ("데이터셋 믹스 (KMMLU 등)",        "실패 (하락)"),
    ("보정 데이터 샘플 수 확대",         "512가 최적"),
    ("Pruning",                        "실패"),
    ("취약 레이어 탐색",                "20번째부터 에러 급등"),
    ("정밀도 조정 + actorder",          "라이브러리 미지원"),
    ("그룹 사이즈 64/32",              "128이 최적"),
    ("v/o 프로젝션 개별 압축",          "효과 미미"),
    ("다운 프로젝션 보호",              "전체 보호가 더 효과적"),
    ("헤세행렬 + 층별 샌드위치 보호",    "유의미한 개선 없음"),
]


# 출력

if __name__ == "__main__":
    print("=" * 72)
    print("  EXAONE 4.0 1.2B Quantization — Experiment Summary")
    print("=" * 72)

    # 점수 추이
    print("\n📊 점수 추이")
    print("-" * 72)
    print(f"  {'버전':<12} {'점수':>8} {'시간':>10}  {'설명'}")
    print("-" * 72)

    for e in EXPERIMENTS:
        score = f"{e['score']:.4f}" if isinstance(e["score"], (int, float)) else "  N/A "
        mark = " ★" if "★" in e["ver"] else "  "
        print(f"{mark}{e['ver']:<12} {score:>8} {e['time']:>10}  {e['desc']}")

    # 최적 설정
    print("\n\n최적 설정 조합")
    print("-" * 72)
    print("  scheme:          W4A16 GPTQ")
    print("  block_size:      128")
    print("  dampening_frac:  0.01")
    print("  MLP 보호:        re:model\\.layers\\.\\d+\\.mlp\\..*")
    print("  samples:         512")
    print("  seq_length:      512")
    print("  dataset:         LGAI-EXAONE/MANTA-1M")
    print("  seed:            42")

    # 체크리스트
    print("\n\n시도 체크리스트")
    print("-" * 72)
    for task, result in CHECKLIST:
        print(f" {task:<30} → {result}")

    print("\n" + "=" * 72)
