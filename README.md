# 📊 Dự án Phân tích doanh thu bán hàng
### Bài tập lớn Chuyên đề - Nhóm 10

## 👥 Thành viên và Phân công nhiệm vụ
| STT | Thành viên | Nhiệm vụ chính |
|---|---|---|
| 1 | **Phạm Minh Tuyển** (Trưởng nhóm) | Quản lý tiến độ, xử lý file đa bảng mã, lập trình bộ lọc làm sạch dữ liệu. |
| 2 | **Bùi Đình Việt Thắng** | Phụ trách Dataset, kiểm tra tính hợp lệ dữ liệu và phân tích logic doanh thu. |
| 3 | **Đỗ Viết Tuấn** | Lập trình mô hình dự báo (Linear Regression) và tính toán chỉ số đo lường. |
| 4 | **Nguyễn Hoài Nam** | Thiết kế giao diện Dashboard Streamlit và lập trình các biểu đồ trực quan. |
| 5 | **Hà Quang Hà** | Tiền xử lý dữ liệu thô, kiểm thử ứng dụng và viết hướng dẫn sử dụng. |

## 🚀 Công nghệ sử dụng
* **Ngôn ngữ:** Python
* [cite_start]**Giao diện:** Streamlit [cite: 11]
* [cite_start]**Thư viện:** Pandas, Matplotlib, Seaborn 

## 🛠 Hướng dẫn chạy ứng dụng
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install streamlit pandas matplotlib seaborn
2. Khởi chạy ứng dụng:

Bash
streamlit run main.py
📂 Cấu trúc dự án
main.py: File chạy chính của ứng dụng.


data_cleaner.py: Xử lý và làm sạch dữ liệu.


visualizer.py: Chứa các hàm vẽ biểu đồ trực quan.


predictor.py: Mô hình dự báo doanh số.


### Bước 3: Đẩy file lên GitHub
Sau khi lưu file, bạn mở Terminal và chạy các lệnh sau để cập nhật lên GitHub:
1.  **Thêm file:** `git add README.md`
2.  **Xác nhận:** `git commit -m "Thêm README giới thiệu dự án và phân công nhóm"`
3.  **Đẩy lên:** `git push origin main`

Khi hoàn tất, giao diện GitHub của bạn sẽ tự động hiển thị nội dung này ở trang chủ
