"""
============================================================
  NHÓM PHÂN TÍCH DOANH SỐ BÁN HÀNG – SUPERSTORE DATASET
  ─────────────────────────────────────────────────────────
  THÀNH VIÊN : THẮNG 
  VAI TRÒ    : DATA ENGINEERING
  PHẦN CODE  : Phần 0 – Tải & Khảo sát | Phần 1 – Data Cleaning
  ─────────────────────────────────────────────────────────
==========================================
"""

# ── Thư viện cần thiết ────────────────────────────────────────────────────────
import pandas as pd          # Xử lý dữ liệu bảng (DataFrame)
import numpy as np           # Tính toán số học
import matplotlib            # Thư viện vẽ đồ thị
matplotlib.use("Agg")        # Backend không cần màn hình (lưu file thay vì hiển thị cửa sổ)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns        # Vẽ đồ thị thống kê đẹp hơn
import warnings
import tkinter as tk         # Tạo hộp thoại chọn file
from tkinter import filedialog

warnings.filterwarnings("ignore")   # Bỏ qua các cảnh báo không quan trọng
sns.set_theme(style="whitegrid")    # Giao diện biểu đồ có lưới mờ

# ─── Cấu hình chung cho toàn bộ biểu đồ ──────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"     : 130,   # Độ phân giải ảnh xuất ra
    "axes.titlesize" : 13,    # Cỡ chữ tiêu đề biểu đồ
    "axes.labelsize" : 11,    # Cỡ chữ nhãn trục
    "xtick.labelsize": 9,     # Cỡ chữ nhãn trục X
    "ytick.labelsize": 9,     # Cỡ chữ nhãn trục Y
    "legend.fontsize": 9,     # Cỡ chữ chú thích
})

# Bảng màu dùng thống nhất cho toàn dự án
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 0 – TẢI & KHẢO SÁT BAN ĐẦU
# Mục tiêu: Nạp dữ liệu vào bộ nhớ và xem tổng quan cấu trúc dataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 0: TẢI DỮ LIỆU & KHẢO SÁT BAN ĐẦU")
print("=" * 62)

# Tạo cửa sổ tkinter ẩn để hiển thị hộp thoại chọn file
root = tk.Tk()
root.withdraw()                      # Ẩn cửa sổ chính tkinter
root.attributes('-topmost', True)    # Hộp thoại luôn hiện trên cùng màn hình

# Mở hộp thoại cho người dùng chọn file CSV
FILE_PATH = filedialog.askopenfilename(
    title="Chọn file dữ liệu CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

# Kiểm tra người dùng có chọn file không
if FILE_PATH:
    df_raw = pd.read_csv(FILE_PATH)
    print(f"Đã tải thành công: {FILE_PATH}")
else:
    print("Bạn chưa chọn file nào! Chương trình sẽ dừng.")
    exit()   # Thoát an toàn nếu không chọn file

# Hiển thị thông tin tổng quan dataset
print(f"\n► Số dòng   : {df_raw.shape[0]:,}")          # Số dòng (records)
print(f"► Số cột    : {df_raw.shape[1]}")              # Số cột (features)
print(f"► Các cột   : {', '.join(df_raw.columns.tolist())}")

print(f"\n► 5 dòng đầu tiên:")
print(df_raw.head().to_string(index=False))            # Xem mẫu dữ liệu đầu

print(f"\n► Kiểu dữ liệu từng cột:")
print(df_raw.dtypes.to_string())                       # int64, float64, object…

print(f"\n► Thống kê mô tả (số):")
print(df_raw.describe().round(2).to_string())          # count, mean, std, min, max…

print(f"\n► Giá trị NULL theo cột:")
null_counts = df_raw.isnull().sum()
print(null_counts[null_counts >= 0].to_string())       # Hiển thị toàn bộ cột (kể cả cột không có NULL)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 – DATA CLEANING (LÀM SẠCH DỮ LIỆU)
# Mục tiêu: Chuẩn hóa dữ liệu thô thành dataset sạch dùng cho phân tích
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING)")
print("=" * 62)

df = df_raw.copy()   # Giữ nguyên df_raw để so sánh trước/sau nếu cần

# ── 1.1 Chuyển đổi cột ngày sang kiểu datetime ────────────────────────────
# Lý do: dữ liệu CSV lưu ngày dạng chuỗi "YYYY-MM-DD", cần chuyển sang datetime
# để tính toán khoảng thời gian và trích xuất year/month/quarter
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"]  = pd.to_datetime(df["Ship Date"])
print("\n[1.1] Chuyển đổi cột ngày: 'Order Date', 'Ship Date' → datetime ✓")

# ── 1.2 Tạo các cột thời gian phái sinh (Feature Engineering) ─────────────
# Các cột này sẽ dùng trong EDA và Machine Learning
df["Year"]      = df["Order Date"].dt.year         # Năm đặt hàng
df["Month"]     = df["Order Date"].dt.month        # Tháng đặt hàng (1-12)
df["YearMonth"] = df["Order Date"].dt.to_period("M")  # Dạng "2016-11" – dùng cho time series
df["Quarter"]   = df["Order Date"].dt.quarter      # Quý (1-4)
df["Ship_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days  # Số ngày từ đặt → giao
print("[1.2] Tạo cột phái sinh: Year, Month, Quarter, YearMonth, Ship_Days ✓")

# ── 1.3 Kiểm tra giá trị âm/bất thường trong cột số ──────────────────────
# Sales và Quantity không thể âm hoặc bằng 0 trong bối cảnh bán hàng
neg_sales = (df["Sales"] <= 0).sum()
neg_qty   = (df["Quantity"] <= 0).sum()
print(f"[1.3] Kiểm tra Sales <= 0: {neg_sales} dòng | Quantity <= 0: {neg_qty} dòng")

# ── 1.4 Phát hiện và xử lý dòng trùng lặp ────────────────────────────────
dup_count = df.duplicated().sum()
print(f"[1.4] Số dòng trùng lặp hoàn toàn: {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)   # Xóa dòng trùng, giữ lần xuất hiện đầu tiên
    print(f"      → Đã xóa {dup_count} dòng trùng lặp.")
else:
    print("      → Không có dòng trùng lặp.")

# ── 1.5 Chuẩn hóa khoảng trắng trong cột văn bản ─────────────────────────
# Tránh trường hợp " West" ≠ "West" khi groupby
for col in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]:
    df[col] = df[col].str.strip()
print("[1.5] Chuẩn hóa khoảng trắng cột văn bản ✓")

# ── 1.6 Tính toán Profit Margin (%) – chỉ số hiệu quả kinh doanh ──────────
# Profit_Margin = (Profit / Sales) × 100  → đơn vị %
# Làm tròn 2 chữ số thập phân
df["Profit_Margin"] = (df["Profit"] / df["Sales"] * 100).round(2)
print("[1.6] Tạo cột Profit_Margin (%) ✓")

print(f"\n► Dữ liệu sau làm sạch: {df.shape[0]:,} dòng × {df.shape[1]} cột")
print("► Không còn giá trị NULL sau tiền xử lý:", df.isnull().sum().sum())

# ── BIỂU ĐỒ 1: Tóm tắt phân phối sau khi làm sạch ──────────────────────
# Mục đích: Kiểm tra trực quan hình dạng phân phối của 3 biến quan trọng
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BIỂU ĐỒ 1 – Tóm Tắt Dữ Liệu Sau Làm Sạch", fontsize=14, fontweight="bold")

# ── Subplot 1a: Phân phối Doanh thu (Sales) ───────────────────────────────
# Histogram giúp thấy: phân phối lệch phải, đơn hàng nhỏ chiếm đa số
axes[0].hist(df["Sales"], bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].set_title("Phân phối Doanh thu (Sales)")
axes[0].set_xlabel("Sales ($)")
axes[0].set_ylabel("Tần suất")

# Vẽ đường mean và median để so sánh (mean > median → lệch phải)
q1, q3 = df["Sales"].quantile([0.25, 0.75])
axes[0].axvline(df["Sales"].mean(),   color="red",    linestyle="--",
                label=f"Mean=${df['Sales'].mean():.0f}")
axes[0].axvline(df["Sales"].median(), color="orange", linestyle="--",
                label=f"Median=${df['Sales'].median():.0f}")
axes[0].legend()

print("\n[BIỂU ĐỒ 1a] Phân phối Sales:")
print(f"  Mean = ${df['Sales'].mean():.2f} | Median = ${df['Sales'].median():.2f} | Std = ${df['Sales'].std():.2f}")
print(f"  → Phân phối lệch phải rõ rệt: phần lớn đơn hàng nhỏ, một số đơn lớn kéo mean lên")

# ── Subplot 1b: Phân phối Số ngày giao hàng ──────────────────────────────
axes[1].hist(df["Ship_Days"], bins=20, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[1].set_title("Phân phối Số ngày giao hàng")
axes[1].set_xlabel("Ship Days")
axes[1].set_ylabel("Tần suất")
print("\n[BIỂU ĐỒ 1b] Số ngày giao hàng (Ship_Days):")
print(f"  Min={df['Ship_Days'].min()} | Max={df['Ship_Days'].max()} | Mean={df['Ship_Days'].mean():.1f}")
print("  → Hầu hết đơn hàng giao trong 1-7 ngày")

# ── Subplot 1c: Phân phối Tỷ lệ giảm giá ────────────────────────────────
axes[2].hist(df["Discount"], bins=20, color=PALETTE[2], edgecolor="white", alpha=0.85)
axes[2].set_title("Phân phối Tỷ lệ giảm giá")
axes[2].set_xlabel("Discount")
axes[2].set_ylabel("Tần suất")
disc_0 = (df["Discount"] == 0).mean() * 100
print("\n[BIỂU ĐỒ 1c] Phân phối Discount:")
print(f"  {disc_0:.1f}% đơn hàng không được giảm giá | Mean discount = {df['Discount'].mean():.2%}")

plt.tight_layout()
plt.savefig("chart_01_data_cleaning.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_01_data_cleaning.png")

# ── Xuất df đã làm sạch để các phần sau dùng lại ─────────────────────────
# Các file khác sẽ import df từ đây hoặc đọc file cleaned
df.to_csv("df_cleaned.csv", index=False)
print("✓ Đã xuất dataset sạch: df_cleaned.csv")
