import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """\
# Phân tích Dịch chuyển Hạng thẻ và Dự báo Ngủ đông trong Hệ thống Loyalty
Notebook này thực hiện các bước:
1. Tải dataset
2. Huấn luyện mô hình (HMM, XGBoost, Random Forest)
3. Đẩy Dataset và Model lên Hugging Face Hub
"""

code_install = """\
!pip install huggingface_hub datasets xgboost scikit-learn pandas hmmlearn
"""

code_hf_login = """\
from huggingface_hub import notebook_login
notebook_login()
"""

code_dataset = """\
from datasets import Dataset
import pandas as pd
import numpy as np

# Tạo mock dataset tương tự Online Retail II đã được feature engineering
# Trong thực tế, bạn sẽ load file CSV từ quá trình Data Preparation
np.random.seed(42)
n_samples = 1000
data = {
    'CustomerID': [f"CUST_{i}" for i in range(n_samples)],
    'Recency': np.random.randint(1, 30, n_samples),
    'Frequency': np.random.randint(1, 10, n_samples),
    'Monetary': np.random.uniform(10, 1000, n_samples),
    'LoyaltyScore': np.random.uniform(1, 5, n_samples),
    'Hidden_State': np.random.randint(0, 4, n_samples),
    'Dormancy_Label': np.random.randint(0, 2, n_samples),
    'Downgrade_Label': np.random.randint(0, 2, n_samples)
}
df = pd.DataFrame(data)

# Chuyển đổi sang Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)
print(hf_dataset)

# Đẩy dataset lên Hugging Face (Thay đổi 'your-username' thành username của bạn)
# hf_dataset.push_to_hub("your-username/loyalty-behavior-dataset")
"""

code_model = """\
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from huggingface_hub import HfApi, hf_hub_download
import joblib
import os

# Chuẩn bị dữ liệu
features = ['Recency', 'Frequency', 'Monetary', 'LoyaltyScore', 'Hidden_State']
X = df[features]
y_dormancy = df['Dormancy_Label']
y_downgrade = df['Downgrade_Label']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_dormancy, test_size=0.2, random_state=42)
X_train_dw, X_test_dw, y_train_dw, y_test_dw = train_test_split(X, y_downgrade, test_size=0.2, random_state=42)

# Huấn luyện XGBoost cho dự báo ngủ đông
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train_d, y_train_d)

# Huấn luyện Random Forest cho dự báo rớt hạng
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_dw, y_train_dw)

# Lưu mô hình
joblib.dump(xgb_model, 'xgboost_dormancy.pkl')
joblib.dump(rf_model, 'rf_downgrade.pkl')

print("Models trained and saved locally.")

# Đẩy mô hình lên Hugging Face
# api = HfApi()
# repo_id = "your-username/loyalty-models"
# api.create_repo(repo_id=repo_id, exist_ok=True)
# api.upload_file(
#     path_or_fileobj="xgboost_dormancy.pkl",
#     path_in_repo="xgboost_dormancy.pkl",
#     repo_id=repo_id,
# )
# api.upload_file(
#     path_or_fileobj="rf_downgrade.pkl",
#     path_in_repo="rf_downgrade.pkl",
#     repo_id=repo_id,
# )
# print("Models pushed to Hugging Face Hub successfully!")
"""

code_inference = """\
# Tải lại mô hình từ Hugging Face
# downloaded_model_path = hf_hub_download(repo_id="your-username/loyalty-models", filename="xgboost_dormancy.pkl")
# loaded_model = joblib.load(downloaded_model_path)
# print("Model loaded from Hugging Face!")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_install),
    nbf.v4.new_code_cell(code_hf_login),
    nbf.v4.new_code_cell(code_dataset),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_code_cell(code_inference)
]

with open('/Users/Vinh/Documents/STUDY/data-mining/notebooks/loyalty-notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
