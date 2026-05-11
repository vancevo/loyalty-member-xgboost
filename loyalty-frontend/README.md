# Loyalty Frontend Dashboard

Đây là thư mục chứa mã nguồn giao diện người dùng (User Interface) của ứng dụng **Phân tích Lòng trung thành Khách hàng**. Ứng dụng này được thiết kế theo dạng Dashboard (bảng điều khiển) chuyên nghiệp nhằm cung cấp cái nhìn trực quan nhất về quy trình Khai phá Dữ liệu (Data Mining) và khả năng ứng dụng AI vào thực tế.

## 1. Vai trò của Frontend

Trong hệ thống End-to-End, Frontend đóng vai trò là "mặt tiền" (Presentation Layer), trực tiếp giao tiếp với người dùng (Marketing/Data Analyst). Nó nhận dữ liệu thao tác từ người dùng, gửi yêu cầu (HTTP Requests) tới **FastAPI Backend**, và hiển thị sinh động kết quả thu được.

Frontend hoàn toàn không lưu trữ bất kỳ mô hình học máy (Machine Learning) hay cơ sở dữ liệu (Database) nào. Mọi sức mạnh xử lý đều được gọi thông qua Backend.

---

## 2. Công nghệ sử dụng

- **Framework**: [Next.js 15](https://nextjs.org/) (App Router). Cung cấp môi trường render siêu tốc và tối ưu hóa SEO tự động.
- **Thư viện UI**: [React 19](https://react.dev/).
- **Styling**: Sử dụng 100% Vanilla CSS (tệp `globals.css` và CSS cục bộ) kết hợp hiệu ứng Glassmorphism (Giao diện trong suốt, hiện đại) thay vì các thư viện CSS nặng nề.

---

## 3. Chức năng chính (Các Tab giao diện)

Hệ thống được chia thành 5 phân hệ hiển thị dưới dạng các Tab độc lập:

1. 🔮 **Dự báo rủi ro (Risk Prediction)**: 
   - Điểm nhấn chính của ứng dụng. Cho phép nhập `Customer ID` (và Tên tuỳ chọn).
   - Truy xuất Real-time vào Backend để tính toán và biểu diễn Xác suất Ngủ đông (Dormancy) & Rớt hạng (Downgrade) thông qua thanh Progress Bar. Hiển thị "Hành động Đề xuất" bằng chữ.

2. ⚙️ **Quy trình Data Mining (Pipeline Presentation)**:
   - Mô phỏng trực quan 10 bước Data Mining của bài tiểu luận thông qua giao diện Accordion.
   - Khi click vào mỗi bước, hiển thị đoạn mã (Sample Code) bằng Python/Pandas mô phỏng thiết kế trong môi trường Mac/VSCode Editor hiện đại.

3. 📂 **Dữ liệu thô (Raw Data Lookup)**:
   - Một bảng (Table) kết nối với Backend để duyệt và tìm kiếm tập dữ liệu khách hàng ban đầu (gồm Customer ID, Tên, Quốc gia, Tổng tiền).

4. 📊 **Dataset Explorer**:
   - Giao diện đồng bộ hoá với **Hugging Face** thông qua Backend. Hiển thị bảng dữ liệu 22 đặc trưng (Rolling RFM, Momentum, Loyalty Score, HMM State) đã qua chế biến, sẵn sàng cho Machine Learning.

5. ℹ️ **Tổng quan hệ thống (System Architecture)**:
   - Bảng thông tin vắn tắt tóm lược thông số cấu hình, thuật toán, nền tảng (XGBoost, Random Forest, Vercel, Docker...).

---

## 4. Hướng dẫn Cài đặt & Chạy ứng dụng

Bạn cần cài đặt **Node.js** (phiên bản 18+).

**Bước 1: Cài đặt thư viện**
Mở terminal và trỏ vào thư mục `loyalty-frontend`:
```bash
npm install
```

**Bước 2: Cấu hình biến môi trường (Tuỳ chọn)**
Mặc định hệ thống trỏ về Backend đang chạy nội bộ ở cổng 8000. Nếu Backend chạy ở nơi khác, hãy tạo tệp `.env.local` ở thư mục `loyalty-frontend`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Bước 3: Chạy chế độ Development**
```bash
npm run dev
```
Giao diện sẽ có sẵn tại **[http://localhost:3000](http://localhost:3000)**. Bất kỳ thay đổi nào trong tệp `src/app/page.js` hay `globals.css` đều được Hot-reload hiển thị ngay lập tức!

**Bước 4: Build cho Production (Môi trường thực tế)**
```bash
npm run build
npm run start
```

---

## 5. Cấu trúc thư mục

- `src/app/page.js`: Nơi chứa toàn bộ mã nguồn React logic chính (Component Tabs, State, API Fetching).
- `src/app/globals.css`: Nơi định nghĩa hệ thống biến màu (CSS Variables), Reset CSS và toàn bộ bộ nhận diện thương hiệu hình ảnh (Glassmorphism, Gradient).
- `src/app/layout.js`: Tệp gốc quy định cấu trúc thẻ HTML và Font chữ (Inter).
