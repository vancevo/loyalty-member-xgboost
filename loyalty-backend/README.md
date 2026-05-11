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
- `requirements.txt`: Chứa danh sách các thư viện Python cần thiết (`fastapi`, `xgboost`, `hmmlearn`, `pandas`, `datasets`,...).
- `Dockerfile` & `vercel.json`: Các tệp dùng để đóng gói ứng dụng (Containerization) hoặc cấu hình triển khai hệ thống không máy chủ (Serverless Vercel).
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
2. **Kéo dữ liệu (Pull):** Tại sự kiện Lifespan, Backend kết nối với Hugging Face và tải toàn bộ kho `loyalty-behavior-dataset` cùng các file `.pkl` của `loyalty-models` trực tiếp vào RAM.
3. **Chờ tín hiệu:** API lắng nghe tại các endpoint (như `/api/predict`).
4. **Dự báo rủi ro:** Khi nhận được `Customer_ID` từ Client truyền sang, Backend tra cứu toàn bộ RFM Features và HMM State có sẵn của khách hàng đó trong dataset in-memory, sau đó nạp vào XGBoost và Random Forest để sinh ra kết quả tỷ lệ phần trăm (Probability). Kết quả sẽ kèm theo các chỉ định tự động (Ví dụ: "Gửi voucher kích hoạt...").

---

## 4. Hướng dẫn sử dụng môi trường

Để chạy riêng hệ thống Backend (cho mục đích Dev/Test), hãy làm theo các bước sau:

**1. Khởi tạo môi trường (Khuyến nghị)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Đảm bảo cấu hình biến môi trường**
Tạo file `.env` (nếu chưa có) và khai báo:
```env
HF_TOKEN=hf_*************************
HF_USERNAME=vancevo
HF_MODEL_REPO=loyalty-models
HF_DATASET_REPO=loyalty-behavior-dataset
```

**3. Huấn luyện (Chỉ khi cần train lại dữ liệu)**
```bash
python real_train.py
python push_to_hf.py
```

**4. Mở Server API**
```bash
uvicorn api.index:app --reload --port 8000
```
Server sẽ chạy ở cổng `8000`. Bạn có thể truy cập `http://localhost:8000/docs` để xem giao diện SwaggerUI kiểm thử các API Endpoint nhanh chóng!
