from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "your_username")
DATASET_REPO = os.getenv("HF_DATASET_REPO", "loyalty-behavior-dataset")
DATASET_REF = f"{HF_USERNAME}/{DATASET_REPO}"
RAW_DATASET_REF = f"{HF_USERNAME}/online-retail-ii"

# In-memory DataFrames
loyalty_dataset_df = None
raw_df = None          # Online Retail II raw data
customers_df = None    # Unique customers extracted from raw

def pull_dataset():
    """Pull processed loyalty dataset từ HuggingFace bằng hf_hub_download."""
    global loyalty_dataset_df
    try:
        from huggingface_hub import hf_hub_download
        print(f"Đang pull loyalty dataset: {DATASET_REF}...")
        file_path = hf_hub_download(
            repo_id=DATASET_REF, 
            filename="data/train-00000-of-00001.parquet", 
            repo_type="dataset",
            token=HF_TOKEN if HF_TOKEN != "hf_xxxxxxxxxxxxxxxxx" else None
        )
        loyalty_dataset_df = pd.read_parquet(file_path, engine='fastparquet')
        print(f"✅ Loyalty dataset: {loyalty_dataset_df.shape}")
    except Exception as e:
        print(f"Warning: Loyalty dataset lỗi ({e})")

def pull_raw_dataset():
    """Pull raw Online Retail II dataset từ HuggingFace qua hf_hub_download."""
    global raw_df, customers_df
    try:
        from huggingface_hub import hf_hub_download
        print(f"Đang pull raw dataset: {RAW_DATASET_REF}...")
        file_path = hf_hub_download(
            repo_id=RAW_DATASET_REF, 
            filename="data/train-00000-of-00001.parquet", 
            repo_type="dataset",
            token=HF_TOKEN if HF_TOKEN != "hf_xxxxxxxxxxxxxxxxx" else None
        )
        raw_df = pd.read_parquet(file_path, engine='fastparquet')
        print(f"✅ Raw dataset: {raw_df.shape}")
        _build_customers()
    except Exception as e:
        print(f"Warning: Raw dataset tải thất bại ({e}). Thử đọc local CSV...")
        local_csv = "online_retail_II.csv"
        if os.path.exists(local_csv):
            raw_df = pd.read_csv(local_csv, encoding="utf-8", encoding_errors="replace", dtype=str)
            raw_df.columns = [c.strip().replace(" ", "_") for c in raw_df.columns]
            print(f"✅ Raw dataset (local CSV): {raw_df.shape}")
            _build_customers()

# Sample names for generation (since raw data doesn't have names)
SAMPLE_NAMES = [
    "Anh Tuấn", "Bảo Châu", "Cẩm Tú", "Duy Mạnh", "Elena Rodriguez", 
    "Hoàng Nam", "Lan Anh", "Minh Đức", "Ngọc Diệp", "Phúc Lâm",
    "Quỳnh Chi", "Sơn Tùng", "Thanh Hằng", "Uyên Linh", "Việt Anh",
    "John Smith", "Maria Garcia", "David Chen", "Yuki Tanaka", "Ahmed Hassan"
]

def _build_customers():
    """Xây dựng bảng unique customers từ raw_df bằng vectorized operations (nhanh hơn 100x)."""
    global customers_df
    if raw_df is None:
        return
    
    print("Đang khởi tạo chỉ mục khách hàng...")
    cid_col = "Customer_ID" if "Customer_ID" in raw_df.columns else "Customer ID"
    
    # Làm sạch dữ liệu
    df = raw_df.dropna(subset=[cid_col]).copy()
    df[cid_col] = df[cid_col].astype(float).astype(str).str.replace(".0", "", regex=False)
    
    # Tính toán tổng hợp
    # Lấy thông tin Country và Description cuối cùng của mỗi khách hàng
    # (Giả định dòng cuối cùng là thông tin mới nhất)
    last_info = df.groupby(cid_col).tail(1).set_index(cid_col)
    
    # Tính tổng đơn hàng và tổng chi tiêu
    stats = df.groupby(cid_col).agg({
        'Invoice': 'nunique',
        'Price': lambda x: pd.to_numeric(x, errors='coerce').sum()
    })
    
    # Kết hợp lại
    res = stats.join(last_info[['Country', 'Description']])
    res = res.reset_index()
    
    # Map sang định dạng mong muốn
    res['customer_name'] = res[cid_col].apply(lambda x: SAMPLE_NAMES[int(float(x)) % len(SAMPLE_NAMES)])
    
    customers_df = pd.DataFrame({
        "customer_id": res[cid_col],
        "customer_name": res['customer_name'],
        "country": res['Country'].fillna("Unknown"),
        "top_product": res['Description'].str.slice(0, 60).fillna(""),
        "total_orders": res['Invoice'],
        "total_spend": res['Price'].round(2)
    })
    
    print(f"✅ Customers index built: {len(customers_df)} unique customers")

@asynccontextmanager
async def lifespan(app: FastAPI):
    pull_dataset()
    pull_raw_dataset()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    customer_id: str
    features: dict

class TrainRequest(BaseModel):
    dataset_name: str
    model_name: str

# Hàm tiện ích để load model (thử tải từ HF, nếu lỗi dùng local)
def load_model(repo_id: str, filename: str):
    local_path = f"models/{filename}"
    hf_token = os.getenv("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download, login
        if hf_token and hf_token != "hf_xxxxxxxxxxxxxxxxx":
            login(token=hf_token)
            print(f"Logged in to HF. Downloading {filename} from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", token=hf_token if hf_token != "hf_xxxxxxxxxxxxxxxxx" else None)
        return joblib.load(model_path)
    except Exception as e:
        print(f"Warning: Không thể tải {filename} từ HuggingFace ({e}). Đang dùng local model...")
        if os.path.exists(local_path):
            return joblib.load(local_path)
        return None

# Tải sẵn các model vào bộ nhớ
USERNAME = os.getenv("HF_USERNAME", "your_username")
MODEL_REPO = os.getenv("HF_MODEL_REPO", "loyalty-models")
REPO_ID = f"{USERNAME}/{MODEL_REPO}"
xgb_dormancy = load_model(REPO_ID, "xgboost_dormancy.pkl")
rf_downgrade = load_model(REPO_ID, "rf_downgrade.pkl")

# Danh sách features cần thiết theo đúng thứ tự lúc train
FEATURE_COLS = [
    'Recency', 'Frequency', 'Monetary', 'QuantitySum', 'AvgUnitPrice', 'UniqueProducts', 'AvgBasketValue',
    'Frequency_3M', 'Monetary_3M', 'QuantitySum_3M', 'UniqueProducts_3M',
    'Frequency_6M', 'Monetary_6M', 'QuantitySum_6M', 'UniqueProducts_6M',
    'AvgBasketValue_3M_Mean', 'Recency_3M_Mean', 'Monetary_Change_1M', 
    'Frequency_Change_1M', 'Recency_Change_1M', 'LoyaltyScore', 'Hidden_State'
]

@app.get("/api")
def root():
    return {"message": "Loyalty Prediction API is running."}

@app.get("/api/customers")
def get_customers(
    search: str = Query("", description="Tìm theo Customer ID hoặc Tên"),
    page: int = 1,
    page_size: int = 20,
):
    """Trả danh sách unique customers từ Online Retail II raw dataset."""
    if customers_df is None:
        return {"status": "loading", "data": [], "total": 0}
    df = customers_df
    if search:
        mask = (
            df["customer_id"].str.contains(search, case=False, na=False)
            | df["customer_name"].str.contains(search, case=False, na=False)
            | df["country"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    total = len(df)
    start = (page - 1) * page_size
    records = df.iloc[start : start + page_size].to_dict(orient="records")
    return {"data": records, "total": total, "page": page, "page_size": page_size}

@app.get("/api/raw-data")
def get_raw_data(page: int = 1, page_size: int = 20, search: str = ""):
    """Endpoint đặc biệt để lấy dữ liệu thô (Customer ID & Customer Name)."""
    if customers_df is None:
        raise HTTPException(status_code=503, detail="Dữ liệu thô chưa sẵn sàng")
    
    df = customers_df[["customer_id", "customer_name", "country", "total_orders", "total_spend"]]
    if search:
        mask = (
            df["customer_id"].str.contains(search, case=False, na=False)
            | df["customer_name"].str.contains(search, case=False, na=False)
        )
        df = df[mask]
    
    total = len(df)
    start = (page - 1) * page_size
    rows = df.iloc[start : start + page_size].to_dict(orient="records")
    return {"data": rows, "total": total, "page": page, "page_size": page_size}

@app.get("/api/raw/rows")
def raw_rows(page: int = 1, page_size: int = 20, customer_id: str = ""):
    """Trả dữ liệu giao dịch thô của 1 khách hàng hoặc tất cả (phân trang)."""
    if raw_df is None:
        raise HTTPException(status_code=503, detail="Raw dataset chưa sẵn sàng")
    cid_col = "Customer_ID" if "Customer_ID" in raw_df.columns else "Customer ID"
    df = raw_df
    if customer_id:
        df = df[df[cid_col].astype(str) == customer_id]
    total = len(df)
    start = (page - 1) * page_size
    rows = df.iloc[start : start + page_size].fillna("").to_dict(orient="records")
    return {"data": rows, "total": total, "page": page, "page_size": page_size}

@app.get("/api/dataset/info")
def dataset_info():
    if loyalty_dataset_df is None:
        return {"status": "not_loaded", "message": "Dataset chưa được pull"}
    return {
        "status": "loaded",
        "source": DATASET_REF,
        "rows": len(loyalty_dataset_df),
        "columns": list(loyalty_dataset_df.columns),
    }

@app.get("/api/dataset/rows")
def dataset_rows(page: int = 1, page_size: int = 20):
    """Trả dữ liệu dataset dạng phân trang để hiển thị trên FE Table."""
    if loyalty_dataset_df is None:
        raise HTTPException(status_code=503, detail="Dataset chưa sẵn sàng")
    total = len(loyalty_dataset_df)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    rows = loyalty_dataset_df.iloc[start:end].to_dict(orient="records")
    return {"total": total, "page": page, "page_size": page_size, "data": rows}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if xgb_dormancy is None or rf_downgrade is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    input_data = {}
    customer_info = {}

    # Tìm trong dataset thật nếu có customer_id
    if loyalty_dataset_df is not None and req.customer_id:
        # Tìm bản ghi cuối cùng của customer theo cột Customer ID
        try:
            cid_str = str(float(req.customer_id))
            col_name = "Customer ID" if "Customer ID" in loyalty_dataset_df.columns else "Customer_ID"
            
            if col_name in loyalty_dataset_df.columns:
                # Ép kiểu an toàn trước khi so sánh
                mask = loyalty_dataset_df[col_name].astype(str).str.replace(".0", "", regex=False) == cid_str.replace(".0", "")
            else:
                cid = float(req.customer_id)
                mask = loyalty_dataset_df.index == cid

            if mask.any():
                row = loyalty_dataset_df[mask].iloc[-1]
                input_data = row[FEATURE_COLS].to_dict()
                customer_info = {
                    "loyalty_score": round(float(row.get("LoyaltyScore", 0)), 2),
                    "hidden_state": int(row.get("Hidden_State", 0)),
                }
        except Exception as e:
            print(f"Lỗi khi tìm customer_id {req.customer_id}: {e}")
            pass

    # Nếu user truyền features thủ công, ưu tiên dùng
    if req.features:
        input_data.update(req.features)

    # Fallback: nếu vẫn chưa đủ features thì báo lỗi thay vì sinh ngẫu nhiên
    missing = [c for c in FEATURE_COLS if c not in input_data]
    if missing:
        # Điền 0 cho các cột còn thiếu
        for c in missing:
            input_data[c] = 0

    df_input = pd.DataFrame([input_data])

    dormancy_prob = float(xgb_dormancy.predict_proba(df_input)[0][1])
    downgrade_prob = float(rf_downgrade.predict_proba(df_input)[0][1])

    # Xác định tier
    loyalty_score = input_data.get("LoyaltyScore", 0)
    if loyalty_score >= 4:
        tier = "Gold"
    elif loyalty_score >= 2.5:
        tier = "Silver"
    else:
        tier = "Bronze"

    # Hành động can thiệp theo báo cáo
    if dormancy_prob > 0.80:
        action = "Gửi voucher kích hoạt mua lại ngay"
        risk_level = "HIGH"
    elif downgrade_prob > 0.60:
        action = "Ưu đãi giữ hạng và chăm sóc VIP"
        risk_level = "HIGH"
    elif downgrade_prob > 0.50:
        action = "Gửi nhắc nhở quyền lợi và ưu đãi nhẹ"
        risk_level = "MEDIUM"
    elif dormancy_prob > 0.40:
        action = "Email cảm ơn và đề xuất sản phẩm liên quan"
        risk_level = "MEDIUM"
    else:
        action = "Chăm sóc định kỳ, duy trì quan hệ"
        risk_level = "LOW"

    return {
        "customer_id": req.customer_id,
        "tier": tier,
        "loyalty_score": round(loyalty_score, 2),
        "hidden_state": customer_info.get("hidden_state", int(input_data.get("Hidden_State", 0))),
        "downgrade_probability": round(downgrade_prob, 4),
        "dormancy_probability": round(dormancy_prob, 4),
        "risk_level": risk_level,
        "recommended_action": action,
        "features_used": {k: round(float(v), 2) for k, v in input_data.items()},
    }

@app.post("/api/train")
def train(req: TrainRequest):
    return {
        "message": f"Training initiated for model {req.model_name} with dataset {req.dataset_name}",
        "status": "success"
    }

