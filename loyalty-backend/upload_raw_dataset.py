"""
Upload online_retail_II.csv lên HuggingFace Dataset: vancevo/online-retail-ii
Chạy: docker exec loyalty_backend python /app/upload_raw_dataset.py
"""
import os
import pandas as pd
from huggingface_hub import login, HfApi
from datasets import Dataset

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "vancevo")
DATASET_REPO = f"{HF_USERNAME}/online-retail-ii"

if not HF_TOKEN:
    raise ValueError("HF_TOKEN không tìm thấy. Kiểm tra file .env")

print(f"1. Đăng nhập HuggingFace...")
login(token=HF_TOKEN)
api = HfApi()

print(f"2. Đọc CSV...")
df = pd.read_csv("/app/online_retail_II.csv", encoding="utf-8", encoding_errors="replace", dtype=str)
# Chuẩn hóa tên cột
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
print(f"   → {len(df):,} hàng, cột: {df.columns.tolist()}")

print(f"3. Tạo repo HuggingFace Dataset: {DATASET_REPO}...")
api.create_repo(repo_id=DATASET_REPO, exist_ok=True, repo_type="dataset")

print(f"4. Đang push lên HuggingFace...")
hf_ds = Dataset.from_pandas(df, preserve_index=False)
hf_ds.push_to_hub(DATASET_REPO, token=HF_TOKEN)

print(f"\n✅ XONG! Dataset đã có tại: https://huggingface.co/datasets/{DATASET_REPO}")
