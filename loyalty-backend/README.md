# Loyalty Backend

Đây là thư mục chứa toàn bộ lõi xử lý dữ liệu (Data Pipeline), hệ thống học máy (Machine Learning) và API (FastAPI) cho ứng dụng phân tích lòng trung thành của khách hàng.

## 1. Mục đích

- **Data Mining & Training:** Thực thi quy trình 10 bước khai phá dữ liệu, từ file CSV thô đến việc tạo ra các đặc trưng phức tạp (Rolling RFM, HMM States), huấn luyện mô hình XGBoost và Random Forest.
- **Model Deployment & API:** Tạo ra một máy chủ RESTful API bằng FastAPI để tiếp nhận dữ liệu từ giao diện web (Next.js), nạp các mô hình đã lưu từ Hugging Face vào bộ nhớ tạm (In-memory) và trả về dự đoán rủi ro (Ngủ đông/Rớt hạng) theo thời gian thực (Real-time).

---

## 2. Cấu trúc thư mục

- `api/index.py`: Chứa mã nguồn của FastAPI server. Nơi định nghĩa các endpoint (`/api/predict`, `/api/raw-data`, `/api/dataset/...`) và logic nạp models từ Hugging Face lúc server vừa khởi động.
- `real_train.py`: Script quan trọng nhất trong hệ thống. Đây là toàn bộ pipeline thực hiện **10 bước Data Mining**. Kết quả sinh ra sẽ được xuất vào thư mục `models/` ở định dạng Dataset và File pickle (`.pkl`).
- `push_to_hf.py`: Script chịu trách nhiệm tự động tải các tài nguyên từ thư mục `models/` lên tài khoản **Hugging Face** để phục vụ việc chia sẻ cũng như tải về (download) khi triển khai API trên Serverless.
- `models/`: Thư mục lưu trữ tạm thời các file mô hình và dataset sau khi chạy `real_train.py`.
- `requirements.txt`: Chứa danh sách các thư viện Python cho môi trường Production (`fastapi`, `xgboost`, `scikit-learn`, `pandas`, `fastparquet`,...). *Lưu ý: Để chạy `real_train.py` bạn cần cài thêm thư viện `datasets` và `hmmlearn` thủ công.*
- `Dockerfile` & `vercel.json` / `.vercelignore`: Các tệp cấu hình cho hệ thống Serverless Vercel (đã được tối ưu dung lượng).
- `online_retail_II.csv`: File dữ liệu giao dịch gốc (bạn cần phải cung cấp file này tại thư mục hiện tại để chạy quá trình train nội bộ).

---

## 3. Luồng hoạt động (Execution Flow)

Hệ thống hoạt động theo 2 luồng độc lập: Luồng Huấn luyện (Offline) và Luồng Phục vụ (Online/API).

### Luồng A: Huấn luyện Dữ liệu & Đẩy lên Cloud (Data Pipeline Flow)
*Chỉ chạy luồng này khi bạn có dữ liệu mới và muốn cập nhật thuật toán/models.*

1. **Chuẩn bị Dữ liệu:** Script đọc file `online_retail_II.csv` trực tiếp từ ổ cứng để tiết kiệm thời gian.
2. **Khai phá dữ liệu:** Chạy lệnh `python real_train.py`. Script tiến hành gom nhóm khách hàng, tính Rolling RFM, tính Momentum, xếp hạng Tier và train ra HMM States, XGBoost model, RF model.
3. **Lưu trữ cục bộ:** Các mô hình trọng số (`xgboost_dormancy.pkl`, `rf_downgrade.pkl`) và dữ liệu nội suy (`loyalty-dataset`) được lưu vào thư mục `models/`.
4. **Cloud Sync:** Chạy lệnh `python push_to_hf.py`. Script lấy HF_TOKEN từ file `.env` và đẩy toàn bộ `models/` lên Hugging Face. Nhờ vậy mô hình được lưu trữ ở nền tảng đám mây vĩnh viễn.

### Luồng B: Phục vụ Dự đoán (API Service Flow)
*Đây là luồng hoạt động chạy 24/7 của hệ thống phục vụ ứng dụng.*

1. **Khởi động API:** Khi chạy `uvicorn api.index:app`, FastAPI kích hoạt sự kiện vòng đời (Lifespan).
2. **Kéo dữ liệu (Pull):** Tại sự kiện Lifespan, Backend kết nối với API của Hugging Face và sử dụng `fastparquet` để tải siêu tốc toàn bộ kho dataset cùng các file `.pkl` của `loyalty-models` trực tiếp vào RAM.
3. **Chờ tín hiệu:** API lắng nghe tại các endpoint (như `/api/predict`).
4. **Dự báo rủi ro:** Khi nhận được `Customer_ID` từ Client truyền sang, Backend tra cứu toàn bộ RFM Features và HMM State có sẵn của khách hàng đó trong dataset in-memory, sau đó nạp vào XGBoost và Random Forest để sinh ra kết quả.

---

## 4. Hướng dẫn sử dụng môi trường

**1. Khởi tạo môi trường (Khuyến nghị)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Nếu muốn chạy script training (real_train.py), cài thêm:
pip install datasets hmmlearn
```

**2. Đảm bảo cấu hình biến môi trường**
Tạo file `.env` (nếu chưa có) và khai báo:
```env
HF_TOKEN=hf_*************************
HF_USERNAME=vancevo
HF_MODEL_REPO=loyalty-models
HF_DATASET_REPO=loyalty-behavior-dataset
```

**3. Mở Server API (Chạy Local)**
```bash
uvicorn api.index:app --reload --port 8000
```
Truy cập `http://localhost:8000/docs` để xem giao diện SwaggerUI kiểm thử API.

---

## 5. Deployment (Triển khai Serverless & Kaggle)

Do Backend sử dụng các thư viện Machine Learning rất nặng (`xgboost`, `scipy`, `pandas`), hệ thống có 2 giải pháp triển khai:

**A. Triển khai Vercel (Miễn phí, 500MB Limit)**
- Đã được cấu hình file `.vercelignore` cắt giảm tối đa thư viện dư thừa (như `pyarrow` hay `datasets`).
- Phù hợp cho mục đích API dự báo nhanh.

**B. Triển khai Kaggle (Miễn phí, RAM 30GB - Khuyên dùng)**
Do hạn chế bộ nhớ của Vercel (500MB Package), Vercel có thể sẽ lỗi khi build. Kaggle là môi trường hoàn hảo để gánh backend này.
- Mở một Notebook mới trên Kaggle.
- Chạy Cell sau để bật API Tunnel:
```python
!git clone https://github.com/vancevo/loyalty-member-xgboost.git
%cd loyalty-member-xgboost/loyalty-backend
!pip install -r requirements.txt
!pip install pyngrok nest-asyncio

import nest_asyncio, os, uvicorn
from pyngrok import ngrok
nest_asyncio.apply()

os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxx"
os.environ["HF_USERNAME"] = "vancevo"
os.environ["HF_MODEL_REPO"] = "loyalty-models"
os.environ["HF_DATASET_REPO"] = "loyalty-behavior-dataset"

# Kết nối Ngrok
public_url = ngrok.connect(8000)
print(f"🚀 API Backend: {public_url}")

uvicorn.run("api.index:app", host="0.0.0.0", port=8000)
```
Sau đó, dán `public_url` vào biến `NEXT_PUBLIC_API_URL` của thư mục Frontend để giao tiếp!
