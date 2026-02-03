import shutil
import os
from google.colab import files

# 1. 제출용 ZIP 파일 생성 (model 폴더를 포함해야 함)
zip_name = "baseline_submit"
print(f"[INFO] {zip_name}.zip 생성 중...")

# root_dir은 현재 디렉토리(.), base_dir은 압축할 대상 폴더(model)
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir="model",
)

print(f"✅ {zip_name}.zip 생성 완료!")

# 2. 내 컴퓨터로 다운로드 (브라우저 팝업이 뜰 때까지 기다려주세요)
files.download(f"{zip_name}.zip")
