import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datasets import Dataset

print("1. Đang đọc dữ liệu từ file local online_retail_II.csv...")
try:
    df = pd.read_csv("online_retail_II.csv", encoding="utf-8", encoding_errors="replace", dtype=str)
    # Đổi tên cột cho chuẩn với file gốc
    df.columns = [c.strip().replace(" ", "_") if c != "Customer ID" else c for c in df.columns]
    if "Customer_ID" in df.columns:
        df.rename(columns={"Customer_ID": "Customer ID"}, inplace=True)
    
    # Ép kiểu lại cho các cột cần tính toán
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
except Exception as e:
    print(f"Lỗi đọc file: {e}")
    exit(1)

print(f"Dữ liệu gốc: {df.shape}")

print("2. Làm sạch dữ liệu...")
# Lọc bỏ missing Customer ID
df = df.dropna(subset=['Customer ID'])
# Lọc bỏ Canceled (bắt đầu bằng C)
df = df[~df['Invoice'].astype(str).str.startswith('C')]
# Lọc số lượng và giá hợp lệ
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
# Thêm cột
df['TotalSum'] = df['Quantity'] * df['Price']
# Chuyển đổi ngày
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

print("3. Xây dựng Monthly Snapshot và RFM...")
# Tính Monthly Snapshot
monthly = df.groupby(['Customer ID', 'YearMonth']).agg({
    'InvoiceDate': 'max',
    'Invoice': 'nunique',
    'TotalSum': 'sum',
    'Quantity': 'sum',
    'Price': 'mean',
    'StockCode': 'nunique'
}).reset_index()
monthly.rename(columns={
    'Invoice': 'Frequency',
    'TotalSum': 'Monetary',
    'Quantity': 'QuantitySum',
    'Price': 'AvgUnitPrice',
    'StockCode': 'UniqueProducts'
}, inplace=True)

# Tính Recency (tính tại cuối mỗi tháng snapshot)
monthly['MonthEnd'] = monthly['YearMonth'].dt.to_timestamp(how='end')
monthly['Recency'] = (monthly['MonthEnd'] - monthly['InvoiceDate']).dt.days

# Tính AvgBasketValue
monthly['AvgBasketValue'] = monthly['Monetary'] / monthly['Frequency']

print("4. Feature Engineering: Rolling & Momentum...")
monthly = monthly.sort_values(['Customer ID', 'YearMonth'])

# Tạo các features dạng Shift (Tháng trước)
monthly['Prev_Monetary'] = monthly.groupby('Customer ID')['Monetary'].shift(1)
monthly['Prev_Frequency'] = monthly.groupby('Customer ID')['Frequency'].shift(1)
monthly['Prev_Recency'] = monthly.groupby('Customer ID')['Recency'].shift(1)

# Tính Momentum
monthly['Monetary_Change_1M'] = monthly['Monetary'] - monthly['Prev_Monetary'].fillna(0)
monthly['Frequency_Change_1M'] = monthly['Frequency'] - monthly['Prev_Frequency'].fillna(0)
monthly['Recency_Change_1M'] = monthly['Recency'] - monthly['Prev_Recency'].fillna(0)

# Tính Rolling 3M, 6M
monthly['Frequency_3M'] = monthly.groupby('Customer ID')['Frequency'].transform(lambda x: x.rolling(3, min_periods=1).sum())
monthly['Monetary_3M'] = monthly.groupby('Customer ID')['Monetary'].transform(lambda x: x.rolling(3, min_periods=1).sum())
monthly['QuantitySum_3M'] = monthly.groupby('Customer ID')['QuantitySum'].transform(lambda x: x.rolling(3, min_periods=1).sum())
monthly['UniqueProducts_3M'] = monthly.groupby('Customer ID')['UniqueProducts'].transform(lambda x: x.rolling(3, min_periods=1).sum())
monthly['AvgBasketValue_3M_Mean'] = monthly.groupby('Customer ID')['AvgBasketValue'].transform(lambda x: x.rolling(3, min_periods=1).mean())
monthly['Recency_3M_Mean'] = monthly.groupby('Customer ID')['Recency'].transform(lambda x: x.rolling(3, min_periods=1).mean())

monthly['Frequency_6M'] = monthly.groupby('Customer ID')['Frequency'].transform(lambda x: x.rolling(6, min_periods=1).sum())
monthly['Monetary_6M'] = monthly.groupby('Customer ID')['Monetary'].transform(lambda x: x.rolling(6, min_periods=1).sum())
monthly['QuantitySum_6M'] = monthly.groupby('Customer ID')['QuantitySum'].transform(lambda x: x.rolling(6, min_periods=1).sum())
monthly['UniqueProducts_6M'] = monthly.groupby('Customer ID')['UniqueProducts'].transform(lambda x: x.rolling(6, min_periods=1).sum())

monthly.fillna(0, inplace=True)

print("5. Loyalty Tier Simulation...")
monthly['R_Score'] = pd.qcut(monthly['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(float)
monthly['F_Score'] = pd.qcut(monthly['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(float)
monthly['M_Score'] = pd.qcut(monthly['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(float)

monthly['LoyaltyScore'] = 0.4 * monthly['R_Score'] + 0.3 * monthly['F_Score'] + 0.3 * monthly['M_Score']

def assign_tier(score):
    if score >= 4: return 3 # Gold
    elif score >= 2.5: return 2 # Silver
    else: return 1 # Bronze
monthly['LoyaltyTier'] = monthly['LoyaltyScore'].apply(assign_tier)

print("6. Hidden Markov Model (HMM)...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(monthly[['Recency', 'Frequency', 'Monetary']])
hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42)
lengths = monthly.groupby('Customer ID').size().values
hmm_model.fit(rfm_scaled, lengths)
monthly['Hidden_State'] = hmm_model.predict(rfm_scaled, lengths)

print("7. Tạo Label (Dormancy và Downgrade)...")
monthly['Next_YearMonth'] = monthly.groupby('Customer ID')['YearMonth'].shift(-1)
# Tính diff an toàn bằng cách tính khác biệt số tháng
def month_diff(start, end):
    if pd.isna(start) or pd.isna(end): return np.nan
    return (end.year - start.year) * 12 + end.month - start.month

monthly['Next_Month_Diff'] = monthly.apply(lambda row: month_diff(row['YearMonth'], row['Next_YearMonth']), axis=1)
monthly['Dormancy_Label'] = np.where(monthly['Next_Month_Diff'] == 1, 0, 1)

monthly['Next_Tier'] = monthly.groupby('Customer ID')['LoyaltyTier'].shift(-1)
monthly['Downgrade_Label'] = np.where(monthly['Next_Tier'] < monthly['LoyaltyTier'], 1, 0)

print(f"Tổng hợp dataset hoàn chỉnh: {monthly.shape}")

print("8. Huấn luyện Model trên dữ liệu thật...")
features = [
    'Recency', 'Frequency', 'Monetary', 'QuantitySum', 'AvgUnitPrice', 'UniqueProducts', 'AvgBasketValue',
    'Frequency_3M', 'Monetary_3M', 'QuantitySum_3M', 'UniqueProducts_3M',
    'Frequency_6M', 'Monetary_6M', 'QuantitySum_6M', 'UniqueProducts_6M',
    'AvgBasketValue_3M_Mean', 'Recency_3M_Mean', 'Monetary_Change_1M', 
    'Frequency_Change_1M', 'Recency_Change_1M', 'LoyaltyScore', 'Hidden_State'
]

X = monthly[features]
y_dormancy = monthly['Dormancy_Label']
y_downgrade = monthly['Downgrade_Label']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_dormancy, test_size=0.2, random_state=42)
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train_d, y_train_d)

X_train_dw, X_test_dw, y_train_dw, y_test_dw = train_test_split(X, y_downgrade, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_dw, y_train_dw)

os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, 'models/xgboost_dormancy.pkl')
joblib.dump(rf_model, 'models/rf_downgrade.pkl')

print("9. Lưu Model và Dataset thành công!")
# Chuyển int, float numpy về standard types cho HuggingFace
export_df = monthly[['Customer ID'] + features + ['Dormancy_Label', 'Downgrade_Label']].copy()
for col in features + ['Dormancy_Label', 'Downgrade_Label']:
    export_df[col] = export_df[col].astype(float)
export_df['Customer ID'] = export_df['Customer ID'].astype(str)

hf_dataset = Dataset.from_pandas(export_df)
hf_dataset.save_to_disk("models/loyalty-dataset")
print("\n🔥 KẾT THÚC! Bạn có thể chạy lại `push_to_hf.py` để đẩy dataset thật lên Hugging Face!")
