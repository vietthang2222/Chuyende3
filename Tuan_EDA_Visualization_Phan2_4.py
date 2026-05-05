"""
============================================================
  NHÓM PHÂN TÍCH DOANH SỐ BÁN HÀNG – SUPERSTORE DATASET
  ─────────────────────────────────────────────────────────
  THÀNH VIÊN : TUẤN
  VAI TRÒ    : EDA + VISUALIZATION + ML CHART
  PHẦN CODE  : Phần 2 – EDA | Phần 4 – Sản phẩm | Biểu đồ ML

"""

# ── Thư viện ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            # Lưu file PNG, không mở cửa sổ
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ── Đọc dataset đã làm sạch từ file Thắng xuất ra ────────────────────────────
# Nếu chạy độc lập, đọc từ file CSV đã clean
df = pd.read_csv("df_cleaned.csv", parse_dates=["Order Date", "Ship Date"])

# Khôi phục cột YearMonth dạng Period (bị mất khi lưu CSV)
df["YearMonth"] = pd.to_datetime(df["Order Date"]).dt.to_period("M")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 – PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
# Mục tiêu: Hiểu phân bố doanh thu và lợi nhuận theo các chiều phân tích
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)")
print("=" * 62)

# ── 2.1 Tổng hợp doanh thu theo 3 chiều chính ────────────────────────────
# groupby + sum: cộng dồn Sales theo từng nhóm, rồi sắp xếp giảm dần
cat_sales    = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
seg_sales    = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)

print("\n[2.1] Tổng doanh thu theo Danh mục (Category):")
for k, v in cat_sales.items():
    pct = v / cat_sales.sum() * 100    # Tỷ trọng phần trăm mỗi danh mục
    print(f"  {k:<20} ${v:>12,.0f}  ({pct:.1f}%)")

print("\n[2.1] Tổng doanh thu theo Phân khúc (Segment):")
for k, v in seg_sales.items():
    pct = v / seg_sales.sum() * 100
    print(f"  {k:<20} ${v:>12,.0f}  ({pct:.1f}%)")

print("\n[2.1] Tổng doanh thu theo Khu vực (Region):")
for k, v in region_sales.items():
    pct = v / region_sales.sum() * 100
    print(f"  {k:<20} ${v:>12,.0f}  ({pct:.1f}%)")

# ── 2.2 Top sản phẩm bán chạy ────────────────────────────────────────────
top_sub      = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)

print("\n[2.2] Top 10 Sub-Category theo doanh thu:")
for i, (k, v) in enumerate(top_sub.items(), 1):
    print(f"  {i:>2}. {k:<30} ${v:>10,.0f}")

print("\n[2.2] Top 10 sản phẩm cụ thể theo doanh thu:")
for i, (k, v) in enumerate(top_products.items(), 1):
    short = k[:50] + "..." if len(k) > 50 else k   # Rút gọn tên sản phẩm dài
    print(f"  {i:>2}. {short:<53} ${v:>8,.0f}")

# ── 2.3 Phân tích lợi nhuận theo danh mục ────────────────────────────────
cat_profit = df.groupby("Category")[["Sales", "Profit"]].sum()
cat_profit["Margin%"] = (cat_profit["Profit"] / cat_profit["Sales"] * 100).round(1)

print("\n[2.3] Lợi nhuận & Biên lợi nhuận theo Danh mục:")
print(cat_profit.to_string())

# Thống kê đơn hàng bị lỗ
loss_orders = df[df["Profit"] < 0]
print(f"\n[2.3] Đơn hàng lỗ (Profit < 0): {len(loss_orders):,} / {len(df):,} ({len(loss_orders)/len(df)*100:.1f}%)")
print("  Phân bổ đơn lỗ theo Danh mục:")
print(loss_orders.groupby("Category").size().to_string())

# ── BIỂU ĐỒ 2: Tổng quan EDA (6 subplot) ────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig)   # Lưới 2 hàng × 3 cột
fig.suptitle("BIỂU ĐỒ 2 – Phân Tích Khám Phá Dữ Liệu (EDA)", fontsize=14, fontweight="bold")

# ── Subplot 2a: Doanh thu theo Category (cột đứng) ───────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(cat_sales.index, cat_sales.values, color=PALETTE[:3], edgecolor="white")
ax1.set_title("Doanh thu theo Danh mục")
ax1.set_ylabel("Tổng Sales ($)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
# Ghi nhãn giá trị trên đỉnh mỗi cột
for bar, val in zip(bars, cat_sales.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f"${val/1e6:.2f}M", ha="center", fontsize=8, fontweight="bold")
print("\n[BIỂU ĐỒ 2a] → Technology dẫn đầu doanh thu, tiếp theo là Furniture và Office Supplies")

# ── Subplot 2b: Doanh thu theo Region ────────────────────────────────────
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

# ── Subplot 2c: Doanh thu theo Segment ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(seg_sales.index, seg_sales.values, color=PALETTE[3:6], edgecolor="white")
ax3.set_title("Doanh thu theo Phân khúc KH")
ax3.set_ylabel("Tổng Sales ($)")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
for i, (idx, val) in enumerate(seg_sales.items()):
    ax3.text(i, val + 3000, f"${val/1e6:.2f}M", ha="center", fontsize=9, fontweight="bold")
print("[BIỂU ĐỒ 2c] → Phân khúc Consumer đóng góp doanh thu lớn nhất (~50%)")

# ── Subplot 2d: Top 10 Sub-Category (cột ngang) ───────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
# Màu đặc biệt cho Sub-Category đứng đầu (PALETTE[0]) và các Sub khác (PALETTE[1])
colors_bar = [PALETTE[0] if i == 0 else PALETTE[1] for i in range(len(top_sub))]
ax4.barh(top_sub.index[::-1], top_sub.values[::-1], color=colors_bar[::-1], edgecolor="white")
ax4.set_title("Top 10 Sub-Category theo Doanh thu")
ax4.set_xlabel("Tổng Sales ($)")
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
print("[BIỂU ĐỒ 2d] → Phones và Chairs là 2 sub-category dẫn đầu doanh thu")

# ── Subplot 2e: Bubble chart Sales vs Profit theo Category ────────────────
ax5 = fig.add_subplot(gs[1, 1])
cat_colors = {"Technology": PALETTE[0], "Furniture": PALETTE[2], "Office Supplies": PALETTE[1]}
for cat, row in cat_profit.iterrows():
    # Kích thước bubble tỷ lệ với |Profit Margin|, tối thiểu 50 để thấy được
    ax5.scatter(row["Sales"], row["Profit"],
                s=max(abs(row["Margin%"]) * 30, 50),
                color=cat_colors.get(cat, "gray"), label=cat,
                alpha=0.9, edgecolors="white", linewidth=1.5)
    ax5.annotate(f'{cat}\n({row["Margin%"]}%)', (row["Sales"], row["Profit"]),
                 textcoords="offset points", xytext=(8, 0), fontsize=8)
ax5.set_title("Sales vs Profit theo Danh mục")
ax5.set_xlabel("Tổng Sales ($)")
ax5.set_ylabel("Tổng Profit ($)")
ax5.legend()
print("[BIỂU ĐỒ 2e] → Office Supplies có biên lợi nhuận % cao nhất; Furniture rất thấp")

# ── Subplot 2f: Phân phối Profit Margin ──────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
# clip(-100, 100): cắt bớt outlier để biểu đồ dễ đọc
ax6.hist(df["Profit_Margin"].clip(-100, 100), bins=50,
         color=PALETTE[4], edgecolor="white", alpha=0.8)
ax6.axvline(0, color="red",   linestyle="--", linewidth=1.5, label="Breakeven")
ax6.axvline(df["Profit_Margin"].mean(), color="green", linestyle="--", linewidth=1.5,
            label=f"Mean={df['Profit_Margin'].mean():.1f}%")
ax6.set_title("Phân phối Profit Margin (%)")
ax6.set_xlabel("Profit Margin (%)")
ax6.set_ylabel("Tần suất")
ax6.legend()
print("[BIỂU ĐỒ 2f] → Phần lớn đơn hàng có biên lợi nhuận dương, nhưng có đáng kể đơn lỗ")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig("chart_02_eda.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_02_eda.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 – SẢN PHẨM BÁN CHẠY & PHÂN TÍCH DANH MỤC
# Mục tiêu: Đi sâu vào từng Sub-Category, tìm sản phẩm lời/lỗ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 4: SẢN PHẨM BÁN CHẠY & PHÂN TÍCH DANH MỤC")
print("=" * 62)

# Top Sub-Category theo số lượng bán ra (Quantity)
top_qty = df.groupby("Sub-Category")["Quantity"].sum().sort_values(ascending=False)
print("\n[4.1] Top Sub-Category theo số lượng bán (Quantity):")
for i, (k, v) in enumerate(top_qty.items(), 1):
    print(f"  {i:>2}. {k:<30} {v:>8,} đơn vị")

# Phân tích tổng hợp đa chiều cho từng Sub-Category
# agg() cho phép tính nhiều chỉ số cùng lúc
sub_analysis = df.groupby("Sub-Category").agg(
    Total_Sales   =("Sales",    "sum"),
    Total_Profit  =("Profit",   "sum"),
    Avg_Discount  =("Discount", "mean"),
    Num_Orders    =("Order ID", "count"),
).assign(
    # Tính Margin sau khi có Sales và Profit
    Margin=lambda x: (x["Total_Profit"] / x["Total_Sales"] * 100).round(1)
).sort_values("Total_Sales", ascending=False)

print("\n[4.2] Phân tích đầy đủ Sub-Category:")
print(f"{'Sub-Category':<25} {'Sales':>12} {'Profit':>10} {'Margin%':>8} {'Discount':>9} {'Orders':>7}")
print("-" * 75)
for idx, row in sub_analysis.iterrows():
    print(f"  {idx:<23} ${row.Total_Sales:>10,.0f} ${row.Total_Profit:>8,.0f} "
          f"{row.Margin:>7.1f}% {row.Avg_Discount:>8.1%} {row.Num_Orders:>6}")

# Lọc Sub-Category có lợi nhuận âm (cần chú ý chiến lược)
loss_sub = sub_analysis[sub_analysis["Total_Profit"] < 0].sort_values("Total_Profit")
print(f"\n[4.3] Sub-Category có lợi nhuận ÂM (cần xem xét chiến lược):")
for idx, row in loss_sub.iterrows():
    print(f"  {idx:<25} Profit = ${row.Total_Profit:>8,.0f} | Margin = {row.Margin:.1f}%")

# ── BIỂU ĐỒ 4: Phân tích sản phẩm (3 subplot) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("BIỂU ĐỒ 4 – Phân Tích Sản Phẩm & Danh Mục", fontsize=14, fontweight="bold")

# ── Subplot 4a: Top Sub-Category theo Sales (màu đỏ nếu lỗ) ──────────────
ax = axes[0]
colors_sc = [PALETTE[3] if sub_analysis.loc[idx, "Total_Profit"] < 0 else PALETTE[0]
             for idx in top_sub.index]
ax.barh(top_sub.index[::-1], top_sub.values[::-1], color=colors_sc[::-1], edgecolor="white")
ax.set_title("Top Sub-Category theo Doanh thu\n(🔴 = lợi nhuận âm)")
ax.set_xlabel("Tổng Sales ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
print("\n[BIỂU ĐỒ 4a] → Phones & Chairs top doanh thu | Tables lỗ nặng nhất")

# ── Subplot 4b: Bubble chart Sales vs Profit (Sub-Category) ──────────────
# Kích thước bubble = số đơn hàng / 5 (chia để không quá to)
ax = axes[1]
for i, (idx, row) in enumerate(sub_analysis.iterrows()):
    color = PALETTE[3] if row.Total_Profit < 0 else PALETTE[0]
    ax.scatter(row.Total_Sales, row.Total_Profit,
               s=abs(row.Num_Orders) / 5, color=color, alpha=0.7, edgecolors="white")
    ax.annotate(idx, (row.Total_Sales, row.Total_Profit), fontsize=6.5, ha="left", alpha=0.85)
ax.axhline(0, color="red", linewidth=1, linestyle="--")   # Đường hòa vốn
ax.set_title("Sales vs Profit (Sub-Category)\nKích thước = Số đơn hàng")
ax.set_xlabel("Total Sales ($)")
ax.set_ylabel("Total Profit ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
print("[BIỂU ĐỒ 4b] → Copiers & Phones có Sales cao và Profit tốt; Tables lỗ dù Sales khá")

# ── Subplot 4c: Profit Margin theo Sub-Category (cột ngang) ──────────────
ax = axes[2]
margin_sorted = sub_analysis["Margin"].sort_values()
colors_m = [PALETTE[3] if v < 0 else PALETTE[1] for v in margin_sorted.values]
ax.barh(margin_sorted.index, margin_sorted.values, color=colors_m, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)   # Đường 0% – ranh giới lời/lỗ
ax.set_title("Profit Margin (%) theo Sub-Category")
ax.set_xlabel("Profit Margin (%)")
# Ghi nhãn phần trăm bên cạnh mỗi thanh
for i, (idx, v) in enumerate(margin_sorted.items()):
    ax.text(v + (0.3 if v >= 0 else -0.5), i, f"{v:.1f}%", va="center", fontsize=7.5)
print("[BIỂU ĐỒ 4c] → Copiers & Labels có biên lợi nhuận cao nhất; Tables & Bookcases âm")

plt.tight_layout()
plt.savefig("chart_04_product.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_04_product.png")


# ══════════════════════════════════════════════════════════════════════════════
# BIỂU ĐỒ ML – ACTUAL vs PREDICTED & FEATURE IMPORTANCE
# Phần này vẽ lại biểu đồ cho kết quả mô hình ML (dữ liệu nhận từ Tuyển & Hà)
# Trong pipeline thực, Tuấn nhận y_test, y_pred_lr, y_pred_rf, feat_imp từ file Tuyển
# ══════════════════════════════════════════════════════════════════════════════
print("\n[THÔNG BÁO] Biểu đồ ML (7A, 7B) được vẽ trong file Tuyển và Hà")
print("  Tuấn phụ trách vẽ lại/tinh chỉnh style sau khi nhận số liệu từ 2 bạn đó.")

# ── Xuất sub_analysis để Nam dùng ─────────────────────────────────────────
sub_analysis.to_csv("sub_analysis.csv")
print("\n✓ Đã xuất sub_analysis.csv cho phần phân tích insight (Nam)")
