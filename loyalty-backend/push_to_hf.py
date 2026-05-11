import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from datasets import load_from_disk

# Nạp biến môi trường từ file .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
USERNAME = os.getenv("HF_USERNAME")
MODEL_REPO_NAME = os.getenv("HF_MODEL_REPO", "loyalty-models")
DATASET_REPO_NAME = os.getenv("HF_DATASET_REPO", "loyalty-behavior-dataset")

if not HF_TOKEN or HF_TOKEN == "hf_xxxxxxxxxxxxxxxxx":
    print("Vui lòng mở file .env và điền HF_TOKEN và HF_USERNAME của bạn!")
    exit(1)

# Login
login(token=HF_TOKEN)
api = HfApi()

# --- A. PUSH MODEL ---
repo_id = f"{USERNAME}/{MODEL_REPO_NAME}"
print(f"1. Đang tạo Model Repo: {repo_id}...")
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

print("Đang upload file xgboost_dormancy.pkl...")
api.upload_file(
    path_or_fileobj="models/xgboost_dormancy.pkl",
    path_in_repo="xgboost_dormancy.pkl",
    repo_id=repo_id,
)

print("Đang upload file rf_downgrade.pkl...")
api.upload_file(
    path_or_fileobj="models/rf_downgrade.pkl",
    path_in_repo="rf_downgrade.pkl",
    repo_id=repo_id,
)
print("✅ PUSH MODEL THÀNH CÔNG!")


# --- B. PUSH DATASET ---
dataset_id = f"{USERNAME}/{DATASET_REPO_NAME}"
print(f"\n2. Đang tạo Dataset Repo: {dataset_id}...")
api.create_repo(repo_id=dataset_id, exist_ok=True, repo_type="dataset")

print("Đang upload Dataset lên Hugging Face...")
hf_dataset = load_from_disk("models/loyalty-dataset")
hf_dataset.push_to_hub(dataset_id, token=HF_TOKEN)

print("✅ PUSH DATASET THÀNH CÔNG!")
print("\n🎉 Tất cả đã được đẩy lên Hugging Face của bạn!")
