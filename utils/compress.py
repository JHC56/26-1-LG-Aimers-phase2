"""
제출용 ZIP 생성 스크립트
Google Colab에서 모델 양자화 후 DACON 제출용 zip 파일을 생성

Usage (Colab):
    !python utils/compress.py
"""

import shutil

zip_name = "baseline_submit"
print(f"[INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir="model",
)

print(f"[완료] {zip_name}.zip 생성 완료")

# Colab 환경에서만 다운로드 실행
try:
    from google.colab import files
    files.download(f"{zip_name}.zip")
except ImportError:
    print(f"[INFO] 로컬 환경: {zip_name}.zip 파일을 직접 확인하세요.")
