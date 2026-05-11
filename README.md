# Khai phá Dữ liệu Lòng trung thành Khách hàng (Loyalty Data Mining)

Dự án này là một hệ thống **End-to-End Data Mining** chuyên sâu, được thiết kế để phân tích hành vi mua sắm, đánh giá lòng trung thành, và dự báo rủi ro (Ngủ đông & Rớt hạng) của khách hàng. Hệ thống bao gồm một quy trình học máy (Machine Learning Pipeline) hoàn chỉnh, từ việc xử lý dữ liệu thô đến việc phục vụ API thời gian thực thông qua kiến trúc Client-Server.

---

## 1. Kiến trúc Hệ thống & Vai trò các Thành phần

Hệ thống được chia làm 3 mảng chính: **Frontend**, **Backend**, và kho lưu trữ đám mây **Hugging Face**.

### 💻 Frontend (Next.js 15)
- **Nhiệm vụ chính:** Giao diện tương tác trực quan (Dashboard) cho phép người dùng (Marketing/Manager) nhập `Customer ID` để dự báo xác suất ngủ đông và rớt hạng.
- **Tính năng nổi bật:** Hiển thị trực tiếp dữ liệu thô (Customer ID & Name), duyệt kho dữ liệu huấn luyện (Dataset Explorer), và trực quan hóa 10 bước Data Mining thông qua mã nguồn mẫu (Sample Code). 
- **Công nghệ:** Next.js (App Router), React 19, CSS Vanilla.

### ⚙️ Backend (FastAPI + Python)
- **Nhiệm vụ chính:** Cung cấp các RESTful API đóng vai trò làm cầu nối giữa giao diện người dùng và các mô hình học máy đã được huấn luyện.
- **Tính năng nổi bật:** 
  - Khi khởi động (Lifespan event), tự động tải bộ Dataset thực tế và các mô hình học máy (XGBoost, Random Forest) từ Hugging Face xuống bộ nhớ tạm.
  - Xử lý các request dự đoán theo thời gian thực (Real-time Inference), tính toán xác suất (Dormancy, Downgrade), và đề xuất hành động marketing can thiệp.
- **Công nghệ:** FastAPI, Uvicorn, Pandas, XGBoost, Scikit-learn.

### ☁️ Kho lưu trữ Hugging Face
- **Nhiệm vụ chính:** Đóng vai trò là trung tâm lưu trữ Model và Dataset (Data Warehouse & Model Registry).
- **Hugging Face Dataset (`vancevo/loyalty-behavior-dataset`):** Lưu trữ bộ dữ liệu đã qua xử lý (RFM, Rolling Windows, HMM states, Customer ID) để dùng làm lookup table khi Backend cần lấy feature tĩnh.
- **Hugging Face Models (`vancevo/loyalty-models`):** Chứa trọng số các mô hình đã huấn luyện (VD: `xgboost_dormancy.pkl`, `rf_downgrade.pkl`).

---

## 2. Quy trình Khai phá Dữ liệu (10 Bước Pipeline)

Quy trình khai phá dữ liệu (`real_train.py`) được thiết kế cẩn thận, giải quyết bài toán chuỗi thời gian (Time-series) của khách hàng:

1. **Thu thập dữ liệu:** Nạp dữ liệu giao dịch *Online Retail II* (hơn 500k dòng).
2. **Làm sạch:** Xóa các bản ghi thiếu `Customer ID`, hủy đơn hàng (Invoice bắt đầu bằng 'C'), hoặc số lượng/giá bán âm.
3. **Chuẩn hóa & Mức khách hàng:** Tính toán tổng doanh thu từng hóa đơn (`Quantity * Price`) và chuẩn hóa thời gian theo tháng (`YearMonth`).
4. **Snapshot hành vi tháng:** Gộp dữ liệu theo từng khách hàng-tháng. Mỗi bản ghi đại diện cho hành vi của một người dùng trong một tháng cụ thể (Tính RFM cơ bản: Recency, Frequency, Monetary).
5. **Trích xuất đặc trưng (Feature Engineering):** Tạo cửa sổ trượt (Rolling 3M/6M) cho các chỉ số RFM để nắm bắt thói quen dài hạn. Tính toán động lượng (Momentum) để phát hiện sự tăng/giảm chi tiêu đột ngột.
6. **Điểm Loyalty & Hạng thẻ:** Dùng thuật toán Quantile (`qcut`) để chia R, F, M thành 5 mức điểm. Tính `Loyalty Score` dựa trên trọng số (R: 40%, F: 30%, M: 30%) và gán hạng thẻ (Gold, Silver, Bronze).
7. **Xây dựng nhãn mục tiêu:** Sử dụng kỹ thuật *Look-ahead 1 tháng* để gán nhãn:
   - `Dormancy_Label` = 1 (Ngủ đông) nếu tháng tiếp theo không phát sinh giao dịch.
   - `Downgrade_Label` = 1 (Rớt hạng) nếu hạng thẻ tháng tiếp theo thấp hơn tháng hiện tại.
8. **Huấn luyện Hidden Markov Model (HMM):** Dùng `GaussianHMM` phân tích chuỗi đặc trưng hành vi qua các tháng để trích xuất 4 "trạng thái ẩn" (Hidden States) (VD: Vip, Thụ động, Ngủ đông...).
9. **Huấn luyện Machine Learning:** 
   - Bài toán Ngủ đông: Đào tạo mô hình **XGBoost** (do xử lý tốt dữ liệu mất cân bằng).
   - Bài toán Rớt hạng: Đào tạo mô hình **Random Forest** phân loại dựa trên 22 biến đặc trưng.
10. **Sinh đầu ra tự động:** Kết xuất các trọng số mô hình dạng `.pkl` và lưu Pandas DataFrame dạng HuggingFace Dataset.

---

## 3. Quá trình Huấn luyện Mô hình (Training)

Nếu bạn muốn tạo lại dữ liệu hoặc huấn luyện lại mô hình từ đầu, bạn sẽ thao tác trên các script Python tại thư mục `loyalty-backend`.

**Bước 1: Chạy file huấn luyện**
```bash
cd loyalty-backend
python real_train.py
```
> **Lưu ý:** Script này đọc dữ liệu tĩnh `online_retail_II.csv`. Nếu chưa có file này, bạn cần tải tập dữ liệu Online Retail II từ kho UCI. Khi chạy xong, script sẽ lưu mô hình XGBoost, Random Forest và Dataset vào thư mục `models/`.

**Bước 2: Đẩy kết quả lên Hugging Face**
```bash
# Đảm bảo bạn đã cấu hình file .env có chứa HF_TOKEN
python push_to_hf.py
```
> Script này sử dụng API của HuggingFace để tự động upload `models/loyalty-dataset` lên HF Datasets và các file `.pkl` lên HF Models. Từ đây, bất kỳ nơi nào chạy Backend cũng có thể lấy lại những phiên bản model mới nhất này.

---

## 4. Hướng dẫn Setup và Cài đặt (Run Locally & Docker)

Bạn có thể chạy dự án thông qua Docker Compose (khuyến nghị) hoặc chạy trực tiếp bằng Terminal.

### 🐳 Cách 1: Chạy toàn bộ hệ thống bằng Docker Compose (Dễ nhất)

1. Hãy chắc chắn bạn đã cài đặt **Docker** và **Docker Compose**.
2. Thiết lập biến môi trường: Tạo file `.env` ở thư mục `loyalty-backend` với nội dung:
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   HF_USERNAME=vancevo
   HF_MODEL_REPO=loyalty-models
   HF_DATASET_REPO=loyalty-behavior-dataset
   ```
3. Khởi động toàn bộ cụm:
   ```bash
   docker-compose up -d --build
   ```
4. Truy cập giao diện: Mở trình duyệt và vào **[http://localhost:3000](http://localhost:3000)**.
*(Backend API sẽ chạy độc lập tại cổng 8000).*

### 💻 Cách 2: Chạy trực tiếp qua Terminal (Dành cho Dev)

**1. Khởi động Backend:**
```bash
cd loyalty-backend
# Khởi tạo môi trường ảo Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Chạy server FastAPI
uvicorn api.index:app --reload --port 8000
```

**2. Khởi động Frontend:**
Mở một terminal mới:
```bash
cd loyalty-frontend
# Cài đặt thư viện Node.js
npm install

# Khởi động ứng dụng Next.js
npm run dev
```
Mở trình duyệt truy cập vào **[http://localhost:3000](http://localhost:3000)**.

---

## 5. Triển khai (Deployment)

Hệ thống được tối ưu hóa để triển khai trên các nền tảng đám mây hiện đại nhằm đảm bảo hiệu năng xử lý Machine Learning:

### 🚀 Backend (Hugging Face Spaces)
Do Backend yêu cầu tài nguyên CPU/RAM lớn để xử lý dữ liệu (hơn 900MB dependencies), chúng tôi sử dụng **Hugging Face Spaces** với SDK **Docker**.
- **Địa chỉ API:** `https://vancevo-loyalty-backend.hf.space`
- **Cách cập nhật:** Sử dụng kỹ thuật "Snapshot Push" để đẩy mã nguồn từ thư mục `loyalty-backend` lên Space (đã được cấu hình trong `Dockerfile` chạy trên cổng 7860).

### 🌐 Frontend (Vercel)
Giao diện Next.js được triển khai trên **Vercel** để đảm bảo tốc độ phản hồi nhanh nhất cho người dùng.
- **Cấu hình:** Biến môi trường `NEXT_PUBLIC_API_URL` trỏ tới địa chỉ API trên Hugging Face Spaces.

---

## 6. Liên hệ & Đóng góp

Nếu bạn có bất kỳ câu hỏi nào về thuật toán HMM hay quy trình huấn luyện XGBoost trong dự án này, vui lòng liên hệ qua trang cá nhân của tác giả.

**Author:** Vinh (Vance Vo)  
**Project:** Loyalty Data Mining System 2026
