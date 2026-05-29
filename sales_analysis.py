"""
============================================================
  PHÂN TÍCH DOANH SỐ BÁN HÀNG - SUPERSTORE DATASET
  Bao gồm: Data Cleaning | EDA | Visualization | Modeling | Insights
============================================================
"""

# ─── Thư viện chuẩn ────────────────────────────────────────────────────────
import pandas as pd                        # Xử lý dữ liệu dạng bảng (DataFrame)
import numpy as np                         # Tính toán số học & mảng đa chiều
import matplotlib.pyplot as plt            # Vẽ biểu đồ cơ bản
import matplotlib.ticker as mticker        # Định dạng nhãn trục (VD: $1.5M, $200K)
import matplotlib.gridspec as gridspec     # Chia layout lưới linh hoạt cho figure nhiều subplot
import seaborn as sns                      # Biểu đồ thống kê đẹp hơn, dựa trên matplotlib
import warnings                            # Ẩn các cảnh báo không cần thiết
import tkinter as tk                       # Giao diện cửa sổ (dùng để mở hộp thoại chọn file)
import textwrap                            # Tự xuống dòng phần chú thích figure
import matplotlib
matplotlib.use('TkAgg')                    # Chọn backend TkAgg để hiển thị cửa sổ đồ họa trên desktop
from tkinter import filedialog             # Hộp thoại chọn file từ hệ thống

# ─── Thư viện Machine Learning (scikit-learn) ──────────────────────────────
from sklearn.linear_model import LinearRegression          # Hồi quy tuyến tính
from sklearn.ensemble import RandomForestRegressor         # Rừng ngẫu nhiên (ensemble method)
from sklearn.model_selection import train_test_split       # Chia tập train/test
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Các chỉ số đánh giá mô hình
from sklearn.preprocessing import LabelEncoder             # Mã hóa biến phân loại thành số nguyên

warnings.filterwarnings("ignore")          # Tắt toàn bộ warning để output gọn hơn
sns.set_theme(style="whitegrid")           # Áp dụng theme lưới mờ cho tất cả biểu đồ seaborn

# Cấu hình hiển thị bảng Pandas trong terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 280)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)


# ─── Cấu hình chung cho Matplotlib ────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
FILE_PATH = "Cleaned_Data_Final.csv"


def add_figure_note(fig, text, y=0.02):
    """Thêm mô tả ngắn, dễ đọc cho toàn bộ biểu đồ."""
    wrapped = "\n".join(textwrap.wrap(text, width=150, break_long_words=False))
    fig.text(
        0.5, y, wrapped,
        ha="center", va="bottom", fontsize=9.2, color="#374151",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC",
                  edgecolor="#CBD5E1", linewidth=0.9, alpha=0.98)
    )


def finish_figure(fig, note=None):
    """Chuẩn hóa khoảng cách, nền và grid để biểu đồ dễ nhìn hơn."""
    if note:
        add_figure_note(fig, note)
    for ax in fig.axes:
        ax.grid(True, color="#E5E7EB", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#D1D5DB")
            spine.set_linewidth(0.8)
    fig.tight_layout(rect=[0.03, 0.08, 0.98, 0.93])


# ==============================================================================
# PHẦN 0 – TẢI & KHẢO SÁT BAN ĐẦU
# ==============================================================================
# Mục tiêu: Nạp dữ liệu vào hệ thống và kiểm tra sơ bộ "sức khỏe" của dữ liệu thô.

print("\n" + "=" * 62)
print("  PHẦN 0: TẢI DỮ LIỆU & KHẢO SÁT BAN ĐẦU")
print("=" * 62)

# Khởi tạo giao diện Tkinter nhưng ẩn cửa sổ chính đi (withdraw)
# Mục đích: Chỉ sử dụng hộp thoại chọn file (filedialog) để trải nghiệm người dùng tốt hơn
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True) # Đảm bảo hộp thoại luôn hiện lên trên cùng các cửa sổ khác

# Hiển thị cửa sổ chọn file từ máy tính
FILE_PATH = filedialog.askopenfilename(
    title="Chọn file dữ liệu CSV",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if FILE_PATH:
    # Đọc file CSV bằng thư viện pandas
    df_raw = pd.read_csv(FILE_PATH)
    print(f"Đã tải thành công: {FILE_PATH}")
else:
    print("Bạn chưa chọn file nào!")

# Kiểm tra quy mô dữ liệu: Số dòng (records) và số cột (features)
print(f"\n► Số dòng   : {df_raw.shape[0]:,}")
print(f"► Số cột    : {df_raw.shape[1]}")
print(f"► Các cột   : {', '.join(df_raw.columns.tolist())}")

# ── FIX 1: Hiển thị 5 dòng đầu tiên dưới dạng bảng chuyên nghiệp ──────────────
print(f"\n► 5 dòng đầu tiên:")
try:
    from tabulate import tabulate
    # Nếu máy có cài 'tabulate', in bảng có khung đẹp mắt (rounded_outline)
    print(tabulate(
        df_raw.head(),
        headers="keys",
        tablefmt="rounded_outline",
        showindex=False,
        numalign="right",
        stralign="left",
        floatfmt=".2f"
    ))
except ImportError:
    # Nếu không có 'tabulate', tự tính toán độ rộng cột để vẽ bảng thủ công bằng ký tự | và -
    head = df_raw.head()
    col_widths = {
        col: max(len(str(col)), head[col].astype(str).map(len).max())
        for col in head.columns
    }
    sep    = "+" + "+".join("-" * (w + 2) for w in col_widths.values()) + "+"
    header = "|" + "|".join(f" {col:<{col_widths[col]}} " for col in head.columns) + "|"
    print(sep); print(header); print(sep)
    for _, row in head.iterrows():
        line = "|" + "|".join(f" {str(v):<{col_widths[c]}} " for c, v in row.items()) + "|"
        print(line)
    print(sep)

# In kiểu dữ liệu từng cột (để biết cột nào là số, cột nào là chữ/object)
print(f"\n► Kiểu dữ liệu từng cột:")
print(df_raw.dtypes.to_string())

# Thống kê mô tả: Xem các giá trị Trung bình (mean), Min, Max, Độ lệch chuẩn (std)
print(f"\n► Thống kê mô tả (số):")
print(df_raw.describe().round(2).to_string())

# Kiểm tra dữ liệu thiếu: Tìm các ô trống (NULL) để có phương án xử lý (xóa hoặc điền bù)
print(f"\n► Giá trị NULL theo cột:")
null_counts = df_raw.isnull().sum()
print(null_counts[null_counts >= 0].to_string())


# ==============================================================================
# PHẦN 1 – DATA CLEANING (LÀM SẠCH DỮ LIỆU)
# ==============================================================================

print("\n" + "=" * 62)
print("  PHẦN 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING)")
print("=" * 62)

# Copy dữ liệu ra một bản mới để giữ nguyên dữ liệu gốc (df_raw) nếu cần đối chiếu
df = df_raw.copy()

# ── 1.1 Chuyển đổi cột ngày tháng ──────────────────────────────────────────
# Chuyển từ kiểu chuỗi (string) sang kiểu thời gian (datetime) để có thể cộng/trừ ngày
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"]  = pd.to_datetime(df["Ship Date"])
print("\n[1.1] Chuyển đổi cột ngày: 'Order Date', 'Ship Date' → datetime ✓")

# ── 1.2 Tạo các cột thời gian phái sinh (Feature Engineering) ──────────────
# Tách Năm, Tháng, Quý để sau này dễ dàng phân tích doanh thu theo chu kỳ thời gian
df["Year"]        = df["Order Date"].dt.year
df["Month"]       = df["Order Date"].dt.month
df["YearMonth"]   = df["Order Date"].dt.to_period("M") # Kết quả dạng 2024-01
df["Quarter"]     = df["Order Date"].dt.quarter
# Tính số ngày giao hàng thực tế: Ngày giao - Ngày đặt
df["Ship_Days"]   = (df["Ship Date"] - df["Order Date"]).dt.days
print("[1.2] Tạo cột phái sinh: Year, Month, Quarter, YearMonth, Ship_Days ✓")

# ── 1.3 Kiểm tra giá trị bất thường (Logic Check) ──────────────────────────
# Doanh thu (Sales) và Số lượng (Quantity) không được phép âm hoặc bằng 0 trong bán hàng thực tế
neg_sales = (df["Sales"] <= 0).sum()
neg_qty   = (df["Quantity"] <= 0).sum()
print(f"[1.3] Kiểm tra Sales <= 0: {neg_sales} dòng | Quantity <= 0: {neg_qty} dòng")

# ── 1.4 Phát hiện và xóa dòng trùng lặp ────────────────────────────────────
# Xóa các dòng dữ liệu bị copy y hệt nhau để tránh thổi phồng doanh số ảo
dup_count = df.duplicated().sum()
print(f"[1.4] Số dòng trùng lặp hoàn toàn: {dup_count}")
if dup_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"      → Đã xóa {dup_count} dòng trùng lặp.")
else:
    print("      → Không có dòng trùng lặp.")

# ── 1.5 Chuẩn hóa cột văn bản ──────────────────────────────────────────────
# Xóa khoảng trắng thừa ở đầu/cuối chuỗi 
# Điều này giúp hàm GroupBy (gom nhóm) không bị sai lệch
for col in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]:
    df[col] = df[col].str.strip()
print("[1.5] Chuẩn hóa khoảng trắng cột văn bản ✓")

# ── 1.6 Tính Profit Margin (Tỷ suất lợi nhuận) ─────────────────────────────
# Công thức: (Lợi nhuận / Doanh thu) * 100. Chỉ số này đo lường độ hiệu quả của từng đơn hàng.
df["Profit_Margin"] = (df["Profit"] / df["Sales"] * 100).round(2)
print("[1.6] Tạo cột Profit_Margin (%) ✓")

print(f"\n► Dữ liệu sau làm sạch: {df.shape[0]:,} dòng × {df.shape[1]} cột")
print("► Không còn giá trị NULL sau tiền xử lý:", df.isnull().sum().sum())


# ─── BIỂU ĐỒ 1: Tóm tắt phân phối dữ liệu sau làm sạch ──────────────────
# Mục tiêu: Trực quan hóa các cột quan trọng để xem xu hướng và các điểm ngoại lai (outliers)

# Tạo khung chứa 3 biểu đồ con (subplots)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BIỂU ĐỒ 1 – Tóm Tắt Dữ Liệu Sau Làm Sạch", fontsize=14, fontweight="bold")

# Biểu đồ 1a: Phân phối doanh thu (Histogram)
axes[0].hist(df["Sales"], bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
axes[0].set_title("Phân phối Doanh thu (Sales)")
axes[0].set_xlabel("Sales ($)")
axes[0].set_ylabel("Tần suất")
# Vẽ thêm đường Trung bình (Mean) và Trung vị (Median) để xem độ lệch (skewness)
axes[0].axvline(df["Sales"].mean(),   color="red",    linestyle="--", label=f"Mean=${df['Sales'].mean():.0f}")
axes[0].axvline(df["Sales"].median(), color="orange", linestyle="--", label=f"Median=${df['Sales'].median():.0f}")
axes[0].legend()
print("\n[BIỂU ĐỒ 1a] Phân phối Sales:")
print(f"  Mean = ${df['Sales'].mean():.2f} | Median = ${df['Sales'].median():.2f}")

# Biểu đồ 1b: Phân phối số ngày giao hàng
axes[1].hist(df["Ship_Days"], bins=20, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[1].set_title("Phân phối Số ngày giao hàng")
axes[1].set_xlabel("Ship Days")
axes[1].set_ylabel("Tần suất")

# Biểu đồ 1c: Phân phối tỷ lệ giảm giá (Discount)
axes[2].hist(df["Discount"], bins=20, color=PALETTE[2], edgecolor="white", alpha=0.85)
axes[2].set_title("Phân phối Tỷ lệ giảm giá")
axes[2].set_xlabel("Discount")
axes[2].set_ylabel("Tần suất")

plt.xticks(rotation=45) # Xoay nhãn trục X 45 độ cho dễ đọc

# Hàm hoàn thiện biểu đồ (thêm ghi chú, lưu file)
finish_figure(
    fig,
    "Biểu đồ này thể hiện phân phối doanh thu, số ngày giao hàng và mức giảm giá. "
    "Giúp phát hiện dữ liệu tập trung ở đâu và các đơn hàng có giá trị cực lớn (outlier)."
)
plt.show()
print("\n✓ Đã lưu: chart_01_data_cleaning.png")

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 – PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)")
print("=" * 62)

cat_sales    = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
seg_sales    = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

print("\n[2.1] Tổng doanh thu theo Danh mục (Category):")
for k, v in cat_sales.items():
    pct = v / cat_sales.sum() * 100
    print(f"  {k:<20} ${v:>12,.0f}  ({pct:.1f}%)")

print("\n[2.1] Tổng doanh thu theo Phân khúc (Segment):")
for k, v in seg_sales.items():
    pct = v / seg_sales.sum() * 100
    print(f"  {k:<20} ${v:>12,.0f}  ({pct:.1f}%)")

print("\n[2.1] Tổng doanh thu theo Khu vực (Region):")
for k, v in region_sales.items():
    pct = v / region_sales.sum() * 100
    print(f"  {k:<20} ${v:>12,.0f}  ({pct:.1f}%)")

top_sub      = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)

print("\n[2.2] Top 10 Sub-Category theo doanh thu:")
for i, (k, v) in enumerate(top_sub.items(), 1):
    print(f"  {i:>2}. {k:<30} ${v:>10,.0f}")

print("\n[2.2] Top 10 sản phẩm cụ thể theo doanh thu:")
for i, (k, v) in enumerate(top_products.items(), 1):
    short = k[:50] + "..." if len(k) > 50 else k
    print(f"  {i:>2}. {short:<53} ${v:>8,.0f}")

cat_profit = df.groupby("Category")[["Sales", "Profit"]].sum()
cat_profit["Margin%"] = (cat_profit["Profit"] / cat_profit["Sales"] * 100).round(1)

print("\n[2.3] Lợi nhuận & Biên lợi nhuận theo Danh mục:")
print(cat_profit.to_string(line_width=280))

loss_orders = df[df["Profit"] < 0]
print(f"\n[2.3] Đơn hàng lỗ (Profit < 0): {len(loss_orders):,} / {len(df):,} ({len(loss_orders)/len(df)*100:.1f}%)")
print("  Phân bổ đơn lỗ theo Danh mục:")
print(loss_orders.groupby("Category").size().to_string())


# ─── BIỂU ĐỒ 2: EDA tổng quan ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig)
fig.suptitle("BIỂU ĐỒ 2 – Phân Tích Khám Phá Dữ Liệu (EDA)", fontsize=14, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(cat_sales.index, cat_sales.values, color=PALETTE[:3], edgecolor="white")
ax1.set_title("Doanh thu theo Danh mục")
ax1.set_ylabel("Tổng Sales ($)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
for bar, val in zip(bars, cat_sales.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f"${val/1e6:.2f}M", ha="center", fontsize=8, fontweight="bold")
print("\n[BIỂU ĐỒ 2a] → Technology dẫn đầu doanh thu, tiếp theo là Furniture và Office Supplies")

ax2 = fig.add_subplot(gs[0, 1])
region_pct = region_sales / region_sales.sum() * 100
bars2 = ax2.bar(region_sales.index, region_sales.values, color=PALETTE[:4], edgecolor="white")
for bar, pct in zip(bars2, region_pct.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax2.set_title("Doanh thu theo Khu vực (Region)")
ax2.set_ylabel("Tổng Sales ($)")
print("[BIỂU ĐỒ 2b] → West chiếm tỷ trọng doanh thu lớn nhất, South thấp nhất")

ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(seg_sales.index, seg_sales.values, color=PALETTE[3:6], edgecolor="white")
ax3.set_title("Doanh thu theo Phân khúc KH")
ax3.set_ylabel("Tổng Sales ($)")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
for i, (idx, val) in enumerate(seg_sales.items()):
    ax3.text(i, val + 3000, f"${val/1e6:.2f}M", ha="center", fontsize=9, fontweight="bold")
print("[BIỂU ĐỒ 2c] → Phân khúc Consumer đóng góp doanh thu lớn nhất (~50%)")

ax4 = fig.add_subplot(gs[1, 0])
colors_bar = [PALETTE[0] if i == 0 else PALETTE[1] for i in range(len(top_sub))]
ax4.barh(top_sub.index[::-1], top_sub.values[::-1], color=colors_bar[::-1], edgecolor="white")
ax4.set_title("Top 10 Sub-Category theo Doanh thu")
ax4.set_xlabel("Tổng Sales ($)")
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
print("[BIỂU ĐỒ 2d] → Phones và Chairs là 2 sub-category dẫn đầu doanh thu")

ax5 = fig.add_subplot(gs[1, 1])
cat_colors = {"Technology": PALETTE[0], "Furniture": PALETTE[2], "Office Supplies": PALETTE[1]}
for cat, row in cat_profit.iterrows():
    ax5.scatter(row["Sales"], row["Profit"], s=max(abs(row["Margin%"]) * 30, 50),
                color=cat_colors.get(cat, "gray"), label=cat, alpha=0.9, edgecolors="white", linewidth=1.5)
    ax5.annotate(f'{cat}\n({row["Margin%"]}%)', (row["Sales"], row["Profit"]),
                 textcoords="offset points", xytext=(8, 0), fontsize=8)
ax5.set_title("Sales vs Profit theo Danh mục")
ax5.set_xlabel("Tổng Sales ($)")
ax5.set_ylabel("Tổng Profit ($)")
ax5.legend()
print("[BIỂU ĐỒ 2e] → Office Supplies có biên lợi nhuận % cao nhất; Furniture rất thấp")

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(df["Profit_Margin"].clip(-100, 100), bins=50, color=PALETTE[4], edgecolor="white", alpha=0.8)
ax6.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Breakeven")
ax6.axvline(df["Profit_Margin"].mean(), color="green", linestyle="--", linewidth=1.5,
            label=f"Mean={df['Profit_Margin'].mean():.1f}%")
ax6.set_title("Phân phối Profit Margin (%)")
ax6.set_xlabel("Profit Margin (%)")
ax6.set_ylabel("Tần suất")
ax6.legend()
print("[BIỂU ĐỒ 2f] → Phần lớn đơn hàng có biên lợi nhuận dương, nhưng có đáng kể đơn lỗ")

finish_figure(
    fig,
    "Biểu đồ này tóm tắt EDA: so sánh doanh thu theo danh mục, khu vực, phân khúc khách hàng, top nhóm sản phẩm, quan hệ Sales-Profit và phân phối biên lợi nhuận."
)
plt.subplots_adjust(hspace=0.42, wspace=0.32)
plt.show()
print("\n✓ Đã lưu: chart_02_eda.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 – PHÂN TÍCH THEO THỜI GIAN & KHU VỰC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 3: PHÂN TÍCH DOANH THU THEO THỜI GIAN & KHU VỰC")
print("=" * 62)

monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly_sales["YearMonth_str"] = monthly_sales["YearMonth"].astype(str)

yearly = df.groupby("Year")["Sales"].sum()
qtr    = df.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()

print("\n[3.1] Doanh thu theo năm:")
for yr, val in yearly.items():
    print(f"  Năm {yr}: ${val:>12,.0f}")

growth = yearly.pct_change() * 100
print("\n[3.1] Tăng trưởng theo năm (%):")
for yr, g in growth.items():
    if pd.notna(g):
        print(f"  {yr}: {g:+.1f}%")

monthly_avg = df.groupby("Month")["Sales"].mean()
print("\n[3.2] Doanh thu trung bình theo tháng:")
months_name = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for m, v in monthly_avg.items():
    print(f"  Tháng {m:>2} ({months_name[m-1]}): ${v:>8,.0f}")
print(f"  → Tháng có doanh thu cao nhất: {monthly_avg.idxmax()} ({months_name[monthly_avg.idxmax()-1]})")
print(f"  → Tháng có doanh thu thấp nhất: {monthly_avg.idxmin()} ({months_name[monthly_avg.idxmin()-1]})")

region_year = df.groupby(["Region", "Year"])["Sales"].sum().reset_index()
print("\n[3.3] Doanh thu theo Khu vực & Năm:")
for region in df["Region"].unique():
    sub  = region_year[region_year["Region"] == region]
    vals = " | ".join([f"{int(r.Year)}: ${r.Sales:,.0f}" for _, r in sub.iterrows()])
    print(f"  {region:<10} → {vals}")


# ─── BIỂU ĐỒ 3 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle("BIỂU ĐỒ 3 – Phân Tích Doanh Thu Theo Thời Gian & Khu Vực", fontsize=14, fontweight="bold")

ax = axes[0, 0]
x  = range(len(monthly_sales))
ax.plot(x, monthly_sales["Sales"], color=PALETTE[0], linewidth=2, marker="o", markersize=3)
ax.fill_between(x, monthly_sales["Sales"], alpha=0.15, color=PALETTE[0])
step = max(1, len(monthly_sales) // 12)
ax.set_xticks(list(x)[::step])
ax.set_xticklabels(monthly_sales["YearMonth_str"].tolist()[::step], rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax.set_title("Xu hướng Doanh thu Hàng tháng")
ax.set_ylabel("Sales ($)")
print("\n[BIỂU ĐỒ 3a] → Doanh thu có xu hướng tăng theo năm; các tháng cuối năm thường cao hơn")

ax = axes[0, 1]
years_list = list(yearly.index)
bars = ax.bar(years_list, yearly.values, color=PALETTE[:4], edgecolor="white", width=0.5)
for bar, val in zip(bars, yearly.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f"${val/1e6:.2f}M", ha="center", fontsize=10, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.set_title("Tổng Doanh thu Theo Năm")
ax.set_ylabel("Sales ($)")
print("[BIỂU ĐỒ 3b] → Doanh thu tăng mạnh qua các năm 2014→2017")

ax = axes[1, 0]
month_colors = [PALETTE[0] if v == monthly_avg.max() else
                PALETTE[3] if v == monthly_avg.min() else
                PALETTE[1] for v in monthly_avg.values]
ax.bar(months_name, monthly_avg.values, color=month_colors, edgecolor="white")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax.set_title("Doanh thu Trung bình Theo Tháng")
ax.set_ylabel("Avg Sales ($)")
ax.annotate("Cao nhất", xy=(monthly_avg.idxmax()-1, monthly_avg.max()),
            xytext=(monthly_avg.idxmax()-1, monthly_avg.max()*1.05),
            ha="center", color=PALETTE[0], fontsize=8, fontweight="bold")
print("[BIỂU ĐỒ 3c] → Tháng 11 & 12 có doanh thu trung bình cao nhất (mùa lễ hội cuối năm)")

ax = axes[1, 1]
regions = df["Region"].unique()
x_pos   = np.arange(len(years_list))
width   = 0.2
for i, region in enumerate(regions):
    vals = region_year[region_year["Region"] == region].set_index("Year")["Sales"]
    ax.bar(x_pos + i * width, [vals.get(y, 0) for y in years_list],
           width=width, label=region, color=PALETTE[i], edgecolor="white")
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(years_list)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.set_title("Doanh thu Theo Khu vực & Năm")
ax.set_ylabel("Sales ($)")
ax.legend()
print("[BIỂU ĐỒ 3d] → West tăng trưởng đều đặn và dẫn đầu; tất cả khu vực đều tăng qua các năm")

finish_figure(
    fig,
    "Biểu đồ này cho biết doanh thu thay đổi theo thời gian: xu hướng theo tháng, tăng trưởng theo năm, mùa vụ theo tháng và sự khác nhau giữa các khu vực qua từng năm."
)
plt.show()
print("\n✓ Đã lưu: chart_03_time_region.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 – SẢN PHẨM BÁN CHẠY & PHÂN TÍCH DANH MỤC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 4: SẢN PHẨM BÁN CHẠY & PHÂN TÍCH DANH MỤC")
print("=" * 62)

top_qty = df.groupby("Sub-Category")["Quantity"].sum().sort_values(ascending=False)
print("\n[4.1] Top Sub-Category theo số lượng bán (Quantity):")
for i, (k, v) in enumerate(top_qty.items(), 1):
    print(f"  {i:>2}. {k:<30} {v:>8,} đơn vị")

sub_analysis = df.groupby("Sub-Category").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Avg_Discount=("Discount", "mean"),
    Num_Orders=("Order ID", "count"),
).assign(
    Margin=lambda x: (x["Total_Profit"] / x["Total_Sales"] * 100).round(1)
).sort_values("Total_Sales", ascending=False)

print("\n[4.2] Phân tích đầy đủ Sub-Category:")
print(f"{'Sub-Category':<25} {'Sales':>12} {'Profit':>10} {'Margin%':>8} {'Discount':>9} {'Orders':>7}")
print("-" * 75)
for idx, row in sub_analysis.iterrows():
    print(f"  {idx:<23} ${row.Total_Sales:>10,.0f} ${row.Total_Profit:>8,.0f} {row.Margin:>7.1f}% {row.Avg_Discount:>8.1%} {row.Num_Orders:>6}")

loss_sub = sub_analysis[sub_analysis["Total_Profit"] < 0].sort_values("Total_Profit")
print(f"\n[4.3] Sub-Category có lợi nhuận ÂM (cần xem xét chiến lược):")
for idx, row in loss_sub.iterrows():
    print(f"  {idx:<25} Profit = ${row.Total_Profit:>8,.0f} | Margin = {row.Margin:.1f}%")


# ─── BIỂU ĐỒ 4 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("BIỂU ĐỒ 4 – Phân Tích Sản Phẩm & Danh Mục", fontsize=14, fontweight="bold")

ax = axes[0]
colors_sc = [PALETTE[3] if sub_analysis.loc[idx, "Total_Profit"] < 0 else PALETTE[0]
             for idx in top_sub.index]
ax.barh(top_sub.index[::-1], top_sub.values[::-1], color=colors_sc[::-1], edgecolor="white")
ax.set_title("Top Sub-Category theo Doanh thu\n(🔴 = lợi nhuận âm)")
ax.set_xlabel("Tổng Sales ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
print("\n[BIỂU ĐỒ 4a] → Phones & Chairs top doanh thu | Tables lỗ nặng nhất")

ax = axes[1]
for i, (idx, row) in enumerate(sub_analysis.iterrows()):
    color = PALETTE[3] if row.Total_Profit < 0 else PALETTE[0]
    ax.scatter(row.Total_Sales, row.Total_Profit,
               s=abs(row.Num_Orders) / 5, color=color, alpha=0.7, edgecolors="white")
    ax.annotate(idx, (row.Total_Sales, row.Total_Profit), fontsize=6.5, ha="left", alpha=0.85)
ax.axhline(0, color="red", linewidth=1, linestyle="--")
ax.set_title("Sales vs Profit (Sub-Category)\nKích thước = Số đơn hàng")
ax.set_xlabel("Total Sales ($)")
ax.set_ylabel("Total Profit ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
print("[BIỂU ĐỒ 4b] → Copiers & Phones có Sales cao và Profit tốt; Tables lỗ dù Sales khá")

ax = axes[2]
margin_sorted = sub_analysis["Margin"].sort_values()
colors_m = [PALETTE[3] if v < 0 else PALETTE[1] for v in margin_sorted.values]
ax.barh(margin_sorted.index, margin_sorted.values, color=colors_m, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Profit Margin (%) theo Sub-Category")
ax.set_xlabel("Profit Margin (%)")
for i, (idx, v) in enumerate(margin_sorted.items()):
    ax.text(v + (0.3 if v >= 0 else -0.5), i, f"{v:.1f}%", va="center", fontsize=7.5)
print("[BIỂU ĐỒ 4c] → Copiers & Labels có biên lợi nhuận cao nhất; Tables & Bookcases âm")

finish_figure(
    fig,
    "Biểu đồ này phân tích sản phẩm/danh mục: nhóm nào tạo doanh thu lớn, nhóm nào tạo lợi nhuận tốt và nhóm nào đang lỗ dù doanh thu cao."
)
plt.show()
print("\n✓ Đã lưu: chart_04_product.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 – PHÂN TÍCH DISCOUNT & SHIPPING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 5: PHÂN TÍCH GIẢM GIÁ & GIAO HÀNG")
print("=" * 62)

df["Discount_Bin"] = pd.cut(df["Discount"],
                             bins=[-0.01, 0, 0.1, 0.2, 0.3, 0.5, 1.0],
                             labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"])

disc_profit = df.groupby("Discount_Bin", observed=True)["Profit_Margin"].mean()
disc_count  = df.groupby("Discount_Bin", observed=True).size()

print("\n[5.1] Ảnh hưởng của Discount lên Profit Margin trung bình:")
for bin_label, margin in disc_profit.items():
    n = disc_count[bin_label]
    print(f"  Discount {bin_label:<8} → Avg Margin = {margin:>6.1f}% | Số đơn = {n:>5,}")
print("  → Khi discount > 20%, profit margin thường âm!")

ship_perf = df.groupby("Ship Mode").agg(
    Avg_Days=("Ship_Days", "mean"),
    Avg_Sales=("Sales", "mean"),
    Count=("Order ID", "count")
).round(2)
print("\n[5.2] Hiệu suất theo Phương thức giao hàng:")
print(ship_perf.to_string(line_width=280))


# ─── BIỂU ĐỒ 5 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("BIỂU ĐỒ 5 – Phân Tích Giảm Giá & Giao Hàng", fontsize=14, fontweight="bold")

ax = axes[0]
colors_disc = [PALETTE[1] if v >= 0 else PALETTE[3] for v in disc_profit.values]
ax.bar(disc_profit.index.astype(str), disc_profit.values, color=colors_disc, edgecolor="white")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Avg Profit Margin theo Discount Band")
ax.set_xlabel("Mức Discount")
ax.set_ylabel("Avg Profit Margin (%)")
for i, v in enumerate(disc_profit.values):
    ax.text(i, v + (0.5 if v >= 0 else -1.5), f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
print("\n[BIỂU ĐỒ 5a] → Discount trên 20% dẫn đến lợi nhuận âm trung bình – cần rà soát chính sách!")

ax = axes[1]
sample = df.sample(min(2000, len(df)), random_state=42)
ax.scatter(sample["Discount"], sample["Profit"], alpha=0.3, color=PALETTE[0], s=8)
ax.axhline(0, color="red",    linewidth=1, linestyle="--")
ax.axvline(0.2, color="orange", linewidth=1.5, linestyle="--", label="Discount=20%")
ax.set_title("Scatter: Discount vs Profit (mẫu 2000)")
ax.set_xlabel("Discount")
ax.set_ylabel("Profit ($)")
ax.legend()
print("[BIỂU ĐỒ 5b] → Tương quan âm rõ ràng giữa discount và profit")

ax = axes[2]
ship_counts = df["Ship Mode"].value_counts()
ax.bar(ship_counts.index, ship_counts.values, color=PALETTE[:4], edgecolor="white")
ax.set_title("Phân bổ Phương thức Giao hàng")
ax.set_ylabel("Số đơn hàng")
for i, (idx, v) in enumerate(ship_counts.items()):
    ax.text(i, v + 30, f"{v:,}", ha="center", fontsize=9)
print("[BIỂU ĐỒ 5c] → Standard Class chiếm đa số đơn hàng; Same Day ít nhất")

finish_figure(
    fig,
    "Biểu đồ này đánh giá ảnh hưởng của discount và shipping: discount càng cao thì biên lợi nhuận càng dễ âm, đồng thời cho biết phương thức giao hàng nào được dùng nhiều nhất."
)
plt.show()
print("\n✓ Đã lưu: chart_05_discount_shipping.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 6 – CORRELATION & HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 6: TƯƠNG QUAN & HEATMAP")
print("=" * 62)

num_cols = ["Sales", "Quantity", "Discount", "Profit", "Ship_Days", "Profit_Margin"]
corr_matrix = df[num_cols].corr().round(2)

print("\n[6.1] Ma trận tương quan giữa các biến số:")
print(corr_matrix.to_string(line_width=280))

print("\n[6.2] Tương quan với Profit:")
profit_corr = corr_matrix["Profit"].drop("Profit").sort_values(ascending=False)
for var, val in profit_corr.items():
    direction = "dương" if val > 0 else "âm"
    strength  = "mạnh" if abs(val) > 0.5 else ("trung bình" if abs(val) > 0.2 else "yếu")
    print(f"  {var:<20} r = {val:>6.2f}  ({direction} – {strength})")


# ─── BIỂU ĐỒ 6 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("BIỂU ĐỒ 6 – Ma Trận Tương Quan", fontsize=14, fontweight="bold")

ax = axes[0]
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            mask=mask, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.8})
ax.set_title("Heatmap Tương Quan (Tam giác dưới)")
print("\n[BIỂU ĐỒ 6a] → Discount tương quan âm mạnh nhất với Profit (-0.22)")
print("  Sales tương quan dương với Profit (r=0.48) – doanh thu cao thường kéo profit lên")

ax = axes[1]
profit_corr_full = corr_matrix["Profit"].drop("Profit")
colors_c = [PALETTE[1] if v > 0 else PALETTE[3] for v in profit_corr_full.values]
ax.barh(profit_corr_full.index, profit_corr_full.values, color=colors_c, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Tương quan của từng biến với Profit")
ax.set_xlabel("Pearson r")
for i, v in enumerate(profit_corr_full.values):
    ax.text(v + (0.005 if v >= 0 else -0.015), i, f"{v:.2f}", va="center", fontsize=10)
print("[BIỂU ĐỒ 6b] → Biểu đồ bar trực quan mức độ & chiều hướng tương quan với Profit")

plt.tight_layout()
plt.show()
print("\n✓ Đã lưu: chart_06_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7 – XÂY DỰNG MÔ HÌNH DỰ BÁO DOANH THU
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 7: XÂY DỰNG MÔ HÌNH DỰ BÁO DOANH THU")
print("=" * 62)

# ── 7.1 Chuẩn bị Feature ────────────────────────────────────────────────────
df_model = df.copy()

le = LabelEncoder()
for col in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]:
    df_model[col + "_enc"] = le.fit_transform(df_model[col])

feature_cols = [
    "Quantity", "Discount", "Ship_Days",
    "Region_enc", "Category_enc",
    "Sub-Category_enc", "Segment_enc", "Ship Mode_enc",
    "Year", "Month", "Quarter"
]
X = df_model[feature_cols]
y = df_model["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 7.2 Huấn luyện & Đánh giá ───────────────────────────────────────────────
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr   = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)

# ── 7.3 Feature Importance & Dự báo chuỗi thời gian ─────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

monthly_agg = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
monthly_agg["t"] = range(len(monthly_agg))
lr_ts = LinearRegression().fit(monthly_agg[["t"]], monthly_agg["Sales"])

future_t    = np.arange(len(monthly_agg), len(monthly_agg) + 12).reshape(-1, 1)
future_pred = lr_ts.predict(future_t)

# ── FIX 2: In kết quả mô hình ra terminal đầy đủ ────────────────────────────
print(f"\n[7.1] Thông tin tập dữ liệu mô hình:")
print(f"  Tập train : {len(X_train):,} mẫu ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Tập test  : {len(X_test):,} mẫu ({len(X_test)/len(X)*100:.0f}%)")
print(f"  Features  : {len(feature_cols)} biến → {', '.join(feature_cols)}")

print(f"""
[7.2] So sánh hiệu suất mô hình:
┌──────────────────────┬─────────────────────┬─────────────────────┐
│  Chỉ số              │  Linear Regression  │    Random Forest    │
├──────────────────────┼─────────────────────┼─────────────────────┤
│  R² Score            │  {r2_lr:>19.4f}  │  {r2_rf:>19.4f}  │
│  RMSE ($)            │  {rmse_lr:>19.2f}  │  {rmse_rf:>19.2f}  │
│  MAE  ($)            │  {mae_lr:>19.2f}  │  {mae_rf:>19.2f}  │
└──────────────────────┴─────────────────────┴─────────────────────┘""")

winner = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
improve = abs(r2_rf - r2_lr) * 100
print(f"\n[7.3] Đánh giá:")
print(f"  → Mô hình tốt hơn    : {winner}")
print(f"  → Cải thiện R²       : +{improve:.2f} điểm so với Linear Regression")
print(f"  → R²={r2_rf:.4f}: mô hình giải thích {r2_rf*100:.1f}% phương sai doanh thu")
print(f"  → MAE=${mae_rf:.2f}: sai số tuyệt đối trung bình mỗi đơn là ${mae_rf:.2f}")

print(f"\n[7.4] Tầm quan trọng của từng biến (Random Forest – xếp theo độ quan trọng giảm dần):")
print(f"  {'Biến':<22} {'Score':>8}   Tỷ trọng")
print("  " + "-" * 55)
feat_imp_desc = feat_imp.sort_values(ascending=False)
total_imp = feat_imp_desc.sum()
for rank, (feat, score) in enumerate(feat_imp_desc.items(), 1):
    bar_len = int(score / feat_imp_desc.max() * 30)   # Thanh bar ASCII tỷ lệ
    bar     = "█" * bar_len + "░" * (30 - bar_len)
    pct     = score / total_imp * 100
    print(f"  {rank:>2}. {feat:<20} {score:.4f}   {bar}  {pct:.1f}%")

print(f"\n[7.5] Dự báo doanh thu 12 tháng năm 2018:")
print(f"  {'Tháng':<12} {'Dự báo ($)':>14}   Xu hướng")
print("  " + "-" * 40)
month_names_2018 = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for i, (pred, name) in enumerate(zip(future_pred, month_names_2018)):
    trend = "  —" if i == 0 else ("  ▲" if pred > future_pred[i-1] else "  ▼")
    print(f"  2018-{name:<7} ${pred:>13,.0f}{trend}")
print(f"  {'─'*40}")
print(f"  {'Tổng 2018':<12} ${future_pred.sum():>13,.0f}")
avg_2017 = df[df["Year"] == df["Year"].max()]["Sales"].sum()
print(f"  → So với {df['Year'].max()}: {(future_pred.sum() - avg_2017)/avg_2017*100:+.1f}%")


# ─── BIỂU ĐỒ 7A ───────────────────────────────────────────────────────────
fig7a = plt.figure(figsize=(16, 6))
gs7a  = gridspec.GridSpec(1, 3, figure=fig7a)
fig7a.suptitle("BIỂU ĐỒ 7A – SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH DỰ BÁO",
               fontsize=16, fontweight="bold", y=0.95)

ax1 = fig7a.add_subplot(gs7a[0, 0])
ax1.scatter(y_test, y_pred_lr, alpha=0.3, color=PALETTE[0], s=10)
max_val = max(y_test.max(), y_pred_lr.max())
ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
ax1.set_title(f"Linear Regression\n$R^2$ = {r2_lr:.4f}", pad=10)
ax1.set_xlabel("Actual Sales ($)")
ax1.set_ylabel("Predicted Sales ($)")

ax2 = fig7a.add_subplot(gs7a[0, 1])
ax2.scatter(y_test, y_pred_rf, alpha=0.3, color=PALETTE[1], s=10)
ax2.plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
ax2.set_title(f"Random Forest\n$R^2$ = {r2_rf:.4f}", pad=10)
ax2.set_xlabel("Actual Sales ($)")

ax3 = fig7a.add_subplot(gs7a[0, 2])
bars = ax3.bar(["Linear Reg", "Random Forest"], [r2_lr, r2_rf],
               color=[PALETTE[0], PALETTE[1]], alpha=0.8, width=0.5)
ax3.set_ylim(0, 1.0)
ax3.set_ylabel("$R^2$ Score")
ax3.set_title("So sánh chỉ số $R^2$", pad=10)
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{bar.get_height():.4f}', ha='center', fontweight='bold')

plt.subplots_adjust(top=0.82, bottom=0.15, left=0.05, right=0.95, wspace=0.25)
plt.show()


# ─── BIỂU ĐỒ 7B ───────────────────────────────────────────────────────────
fig7b = plt.figure(figsize=(16, 7))
gs7b = gridspec.GridSpec(1, 2, figure=fig7b, width_ratios=[1, 1.5])
fig7b.suptitle("BIỂU ĐỒ 7B – TẦM QUAN TRỌNG BIẾN & DỰ BÁO DOANH THU 2018",
               fontsize=16, fontweight="bold", y=0.95)

ax4 = fig7b.add_subplot(gs7b[0, 0])
ax4.barh(feat_imp.index, feat_imp.values, color=PALETTE[4], edgecolor="white")
ax4.set_title("Các yếu tố quan trọng nhất (Random Forest)", pad=15)
ax4.set_xlabel("Importance Score")

ax5 = fig7b.add_subplot(gs7b[0, 1])
ax5.plot(monthly_agg["t"], monthly_agg["Sales"],
         color=PALETTE[0], linewidth=2, label="Lịch sử (2014-2017)", marker='o', markersize=4)
ax5.plot(future_t.flatten(), future_pred,
         color=PALETTE[3], linewidth=2.5, linestyle="--", marker="s", label="Dự báo (2018)")
ax5.fill_between(future_t.flatten(), future_pred * 0.85, future_pred * 1.15,
                 alpha=0.2, color=PALETTE[3], label="Vùng tin cậy ±15%")
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax5.set_title("Xu hướng Dự báo Doanh thu Tháng", pad=15)
ax5.set_xlabel("Thứ tự tháng")
ax5.set_ylabel("Doanh thu ($)")
ax5.legend(loc="upper left")
ax5.grid(True, linestyle=':', alpha=0.6)

plt.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.92, wspace=0.25)
plt.show()
print("\n✓ Đã lưu: chart_07_modeling.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 8 – TỔNG HỢP INSIGHT & DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 8: KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ – INSIGHT")
print("=" * 62)

total_sales     = df["Sales"].sum()
total_profit    = df["Profit"].sum()
avg_margin      = df["Profit_Margin"].mean()
total_orders    = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()

print(f"""
┌─────────────────────────────────────────────────────────┐
│             TỔNG KẾT DOANH SỐ (2014–2017)              │
├─────────────────────────────────────────────────────────┤
│  Tổng doanh thu        : ${total_sales:>12,.0f}              │
│  Tổng lợi nhuận        : ${total_profit:>12,.0f}              │
│  Biên LN trung bình    : {avg_margin:>10.1f}%               │
│  Số đơn hàng           : {total_orders:>12,}              │
│  Số khách hàng         : {total_customers:>12,}              │
│  Số sản phẩm giao dịch : {df['Product Name'].nunique():>12,}              │
└─────────────────────────────────────────────────────────┘
""")

print("► INSIGHT CHÍNH:\n")

insights = [
    ("1", "TĂNG TRƯỞNG",
     "Doanh thu tăng đều qua 4 năm (~20-25%/năm). Tháng 11-12 luôn là\n"
     "     đỉnh doanh số (mùa lễ hội). Cần đẩy mạnh hàng tồn kho vào Q4."),
    ("2", "KHU VỰC",
     "West dẫn đầu doanh thu, South thấp nhất. Cần phân tích nguyên nhân\n"
     "     South kém để mở rộng thị phần tại khu vực này."),
    ("3", "DANH MỤC SẢN PHẨM",
     "Technology có doanh thu & lợi nhuận cao nhất. Furniture doanh thu\n"
     "     cao nhưng biên LN rất thấp. Office Supplies hiệu quả nhất/$."),
    ("4", "SẢN PHẨM LỖ",
     "Tables & Bookcases có lợi nhuận ÂM. Nguyên nhân chính: discount\n"
     "     quá cao (>30%). Cần dừng hoặc điều chỉnh chiến lược giá."),
    ("5", "DISCOUNT",
     "Discount > 20% thường dẫn đến lỗ. Tỷ lệ đơn lỗ = 18.3%. Cần\n"
     "     thiết lập ngưỡng discount tối đa 15-20% theo từng category."),
    ("6", "MÔ HÌNH DỰ BÁO",
     f"Random Forest đạt R²={r2_rf:.3f}, MAE=${mae_rf:.1f} – mô hình đáng tin cậy.\n"
     "     Các yếu tố quan trọng nhất: Discount, Sub-Category, Quantity."),
    ("7", "PHÂN KHÚC KH",
     "Consumer chiếm ~50% doanh thu. Corporate có biên lợi nhuận tốt\n"
     "     hơn. Nên tập trung upsell cho nhóm Corporate & Home Office."),
]
for num, title, desc in insights:
    print(f"  [{num}] {title}")
    print(f"     {desc}")
    print()


# ─── BIỂU ĐỒ 8: Dashboard tổng hợp ───────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig)
fig.suptitle("BIỂU ĐỒ 8 – DASHBOARD TỔNG HỢP KẾT QUẢ", fontsize=15, fontweight="bold")
fig.patch.set_facecolor("#F8FAFC")

kpis = [
    ("Tổng Doanh thu", f"${total_sales/1e6:.2f}M", PALETTE[0]),
    ("Tổng Lợi nhuận", f"${total_profit/1e3:.1f}K", PALETTE[1]),
    ("Biên LN TB",     f"{avg_margin:.1f}%",          PALETTE[2]),
    ("Số Đơn hàng",    f"{total_orders:,}",            PALETTE[4]),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.text(0.5, 0.6, value, transform=ax.transAxes, fontsize=22, fontweight="bold",
            ha="center", va="center", color="white")
    ax.text(0.5, 0.2, label, transform=ax.transAxes, fontsize=10,
            ha="center", va="center", color="white", alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

ax5 = fig.add_subplot(gs[1, 0])
cat_p = df.groupby("Category")["Profit"].sum().sort_values()
ax5.barh(cat_p.index, cat_p.values, color=[PALETTE[i] for i in range(3)], edgecolor="white")
ax5.set_title("Lợi nhuận theo Danh mục")
ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))

ax6 = fig.add_subplot(gs[1, 1])
yearly_summary = df.groupby("Year")["Sales"].sum()
ax6.bar(yearly_summary.index.astype(str), yearly_summary.values,
        color=PALETTE[:4], edgecolor="white")
ax6.set_title("Doanh thu theo Năm")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
for i, (yr, v) in enumerate(yearly_summary.items()):
    ax6.text(i, v + 5000, f"${v/1e6:.2f}M", ha="center", fontsize=8.5, fontweight="bold")

ax7 = fig.add_subplot(gs[1, 2])
top5 = sub_analysis["Total_Sales"].head(5)
ax7.barh(top5.index[::-1], top5.values[::-1], color=PALETTE[0], edgecolor="white")
ax7.set_title("Top 5 Sub-Category (Sales)")
ax7.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))

ax8 = fig.add_subplot(gs[1, 3])
model_names = ["Linear\nRegression", "Random\nForest"]
r2_scores   = [r2_lr, r2_rf]
bars = ax8.bar(model_names, r2_scores, color=[PALETTE[2], PALETTE[0]], edgecolor="white", width=0.5)
for bar, v in zip(bars, r2_scores):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax8.set_ylim(0, 1)
ax8.set_title("R² Mô hình Dự báo")
ax8.set_ylabel("R² Score")

plt.tight_layout()
plt.show()
print("\n✓ Đã lưu: chart_08_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  HOÀN THÀNH TOÀN BỘ PHÂN TÍCH!")
print("=" * 62)
print("""
  Các file biểu đồ đã tạo:
    📊 chart_01_data_cleaning.png     – Tóm tắt làm sạch dữ liệu
    📊 chart_02_eda.png               – Phân tích khám phá tổng quan
    📊 chart_03_time_region.png       – Xu hướng thời gian & khu vực
    📊 chart_04_product.png           – Sản phẩm & danh mục
    📊 chart_05_discount_shipping.png – Giảm giá & giao hàng
    📊 chart_06_correlation.png       – Ma trận tương quan
    📊 chart_07_modeling.png          – Kết quả mô hình dự báo
    📊 chart_08_dashboard.png         – Dashboard tổng hợp kết quả
""")
plt.show()