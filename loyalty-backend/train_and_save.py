import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from datasets import Dataset

# 1. Tạo dataset mô phỏng 22 đặc trưng như trong báo cáo (Rolling RFM, Momentum, v.v.)
np.random.seed(42)
n_samples = 2000

data = {
    'Recency': np.random.randint(1, 90, n_samples),
    'Frequency': np.random.randint(1, 20, n_samples),
    'Monetary': np.random.uniform(50, 5000, n_samples),
    'QuantitySum': np.random.randint(1, 100, n_samples),
    'AvgUnitPrice': np.random.uniform(5, 100, n_samples),
    'UniqueProducts': np.random.randint(1, 20, n_samples),
    'AvgBasketValue': np.random.uniform(10, 500, n_samples),
    'Frequency_3M': np.random.randint(1, 50, n_samples),
    'Monetary_3M': np.random.uniform(100, 10000, n_samples),
    'QuantitySum_3M': np.random.randint(10, 300, n_samples),
    'UniqueProducts_3M': np.random.randint(5, 50, n_samples),
    'Frequency_6M': np.random.randint(1, 100, n_samples),
    'Monetary_6M': np.random.uniform(200, 20000, n_samples),
    'QuantitySum_6M': np.random.randint(20, 600, n_samples),
    'UniqueProducts_6M': np.random.randint(10, 100, n_samples),
    'AvgBasketValue_3M_Mean': np.random.uniform(10, 500, n_samples),
    'Recency_3M_Mean': np.random.randint(1, 90, n_samples),
    'Monetary_Change_1M': np.random.uniform(-1000, 1000, n_samples),
    'Frequency_Change_1M': np.random.randint(-10, 10, n_samples),
    'Recency_Change_1M': np.random.randint(-30, 30, n_samples),
    'LoyaltyScore': np.random.uniform(1, 5, n_samples),
    'Hidden_State': np.random.randint(0, 4, n_samples),
    'Dormancy_Label': np.random.randint(0, 2, n_samples),
    'Downgrade_Label': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)

features = [
    'Recency', 'Frequency', 'Monetary', 'QuantitySum', 'AvgUnitPrice', 'UniqueProducts', 'AvgBasketValue',
    'Frequency_3M', 'Monetary_3M', 'QuantitySum_3M', 'UniqueProducts_3M',
    'Frequency_6M', 'Monetary_6M', 'QuantitySum_6M', 'UniqueProducts_6M',
    'AvgBasketValue_3M_Mean', 'Recency_3M_Mean', 'Monetary_Change_1M', 
    'Frequency_Change_1M', 'Recency_Change_1M', 'LoyaltyScore', 'Hidden_State'
]

X = df[features]
y_dormancy = df['Dormancy_Label']
y_downgrade = df['Downgrade_Label']

# 2. Huấn luyện Model XGBoost cho Dự báo ngủ đông
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_dormancy, test_size=0.2, random_state=42)
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train_d, y_train_d)

# 3. Huấn luyện Model Random Forest cho Dự báo rớt hạng
X_train_dw, X_test_dw, y_train_dw, y_test_dw = train_test_split(X, y_downgrade, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_dw, y_train_dw)

# 4. Lưu Model cục bộ
os.makedirs("loyalty-backend/models", exist_ok=True)
joblib.dump(xgb_model, 'loyalty-backend/models/xgboost_dormancy.pkl')
joblib.dump(rf_model, 'loyalty-backend/models/rf_downgrade.pkl')

print("1. Đã train và lưu model thành công vào loyalty-backend/models/")

# 5. Lưu Dataset
hf_dataset = Dataset.from_pandas(df)
hf_dataset.save_to_disk("loyalty-backend/models/loyalty-dataset")
print("2. Đã lưu dataset thành công")

print("\n--- HƯỚNG DẪN PUSH LÊN HUGGING FACE ---")
print("Để push lên Hugging Face của bạn, hãy cung cấp Token hoặc làm theo README.")
