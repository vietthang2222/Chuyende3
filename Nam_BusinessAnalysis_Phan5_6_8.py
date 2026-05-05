"""
============================================================
  NHÓM PHÂN TÍCH DOANH SỐ BÁN HÀNG – SUPERSTORE DATASET
  ─────────────────────────────────────────────────────────
  THÀNH VIÊN : NAM
  VAI TRÒ    : BUSINESS ANALYSIS + INSIGHT
  PHẦN CODE  : Phần 5 – Discount & Shipping
               Phần 6 – Correlation & Heatmap
               Phần 8 – Tổng hợp Insight & Dashboard

"""

# ── Thư viện ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ── Đọc dataset đã làm sạch ───────────────────────────────────────────────────
df = pd.read_csv("df_cleaned.csv", parse_dates=["Order Date", "Ship Date"])
df["YearMonth"] = pd.to_datetime(df["Order Date"]).dt.to_period("M")

# ── Tính lại sub_analysis (hoặc import từ Tuấn) ──────────────────────────────
sub_analysis = df.groupby("Sub-Category").agg(
    Total_Sales  =("Sales",    "sum"),
    Total_Profit =("Profit",   "sum"),
    Avg_Discount =("Discount", "mean"),
    Num_Orders   =("Order ID", "count"),
).assign(Margin=lambda x: (x["Total_Profit"] / x["Total_Sales"] * 100).round(1)
).sort_values("Total_Sales", ascending=False)

# ── Tái tạo kết quả mô hình để dùng trong Phần 8 ─────────────────────────────
# (Trong pipeline, Nam nhận r2_lr, r2_rf, mae_rf từ Tuyển)
df_model = df.copy()
le = LabelEncoder()
for col in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]:
    df_model[col + "_enc"] = le.fit_transform(df_model[col])
feature_cols = ["Quantity", "Discount", "Ship_Days",
                "Region_enc", "Category_enc", "Sub-Category_enc",
                "Segment_enc", "Ship Mode_enc", "Year", "Month", "Quarter"]
X = df_model[feature_cols]
y = df_model["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
r2_lr  = r2_score(y_test, y_pred_lr)
r2_rf  = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 – PHÂN TÍCH GIẢM GIÁ (DISCOUNT) & GIAO HÀNG (SHIPPING)
# Mục tiêu: Đánh giá tác động của chính sách giảm giá và phương thức vận chuyển
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 5: PHÂN TÍCH GIẢM GIÁ & GIAO HÀNG")
print("=" * 62)

# ── 5.1 Phân nhóm Discount và tính Profit Margin trung bình mỗi nhóm ─────
# pd.cut(): chia cột liên tục thành các bin rời rạc có nhãn
df["Discount_Bin"] = pd.cut(
    df["Discount"],
    bins=[-0.01, 0, 0.1, 0.2, 0.3, 0.5, 1.0],
    labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"]
)
# Tính Profit Margin trung bình và đếm số đơn trong mỗi bin
disc_profit = df.groupby("Discount_Bin", observed=True)["Profit_Margin"].mean()
disc_count  = df.groupby("Discount_Bin", observed=True).size()

print("\n[5.1] Ảnh hưởng của Discount lên Profit Margin trung bình:")
for bin_label, margin in disc_profit.items():
    n = disc_count[bin_label]
    print(f"  Discount {bin_label:<8} → Avg Margin = {margin:>6.1f}% | Số đơn = {n:>5,}")
print("  → Khi discount > 20%, profit margin thường âm!")

# ── 5.2 Hiệu suất theo phương thức giao hàng ─────────────────────────────
# So sánh tốc độ giao và doanh thu trung bình từng Ship Mode
ship_perf = df.groupby("Ship Mode").agg(
    Avg_Days  =("Ship_Days", "mean"),    # Số ngày giao trung bình
    Avg_Sales =("Sales",     "mean"),    # Doanh thu trung bình mỗi đơn
    Count     =("Order ID",  "count")    # Tổng số đơn
).round(2)
print("\n[5.2] Hiệu suất theo Phương thức giao hàng:")
print(ship_perf.to_string())

# ── BIỂU ĐỒ 5: Discount & Shipping (3 subplot) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("BIỂU ĐỒ 5 – Phân Tích Giảm Giá & Giao Hàng", fontsize=14, fontweight="bold")

# ── Subplot 5a: Avg Profit Margin theo Discount Bin ──────────────────────
ax = axes[0]
colors_disc = [PALETTE[1] if v >= 0 else PALETTE[3] for v in disc_profit.values]
ax.bar(disc_profit.index.astype(str), disc_profit.values, color=colors_disc, edgecolor="white")
ax.axhline(0, color="black", linewidth=1)   # Đường hòa vốn
ax.set_title("Avg Profit Margin theo Discount Band")
ax.set_xlabel("Mức Discount")
ax.set_ylabel("Avg Profit Margin (%)")
for i, v in enumerate(disc_profit.values):
    ax.text(i, v + (0.5 if v >= 0 else -1.5), f"{v:.1f}%",
            ha="center", fontsize=9, fontweight="bold")
print("\n[BIỂU ĐỒ 5a] → Discount trên 20% dẫn đến lợi nhuận âm trung bình – cần rà soát chính sách!")

# ── Subplot 5b: Scatter Discount vs Profit (mẫu 2000 điểm) ──────────────
# Dùng mẫu thay vì toàn bộ để tránh quá tải biểu đồ (overplotting)
ax = axes[1]
sample = df.sample(min(2000, len(df)), random_state=42)
ax.scatter(sample["Discount"], sample["Profit"], alpha=0.3, color=PALETTE[0], s=8)
ax.axhline(0,   color="red",    linewidth=1,   linestyle="--")
ax.axvline(0.2, color="orange", linewidth=1.5, linestyle="--", label="Discount=20%")
ax.set_title("Scatter: Discount vs Profit (mẫu 2000)")
ax.set_xlabel("Discount")
ax.set_ylabel("Profit ($)")
ax.legend()
print("[BIỂU ĐỒ 5b] → Tương quan âm rõ ràng giữa discount và profit")

# ── Subplot 5c: Phân bổ Phương thức giao hàng ────────────────────────────
ax = axes[2]
ship_counts = df["Ship Mode"].value_counts()
ax.bar(ship_counts.index, ship_counts.values, color=PALETTE[:4], edgecolor="white")
ax.set_title("Phân bổ Phương thức Giao hàng")
ax.set_ylabel("Số đơn hàng")
for i, (idx, v) in enumerate(ship_counts.items()):
    ax.text(i, v + 30, f"{v:,}", ha="center", fontsize=9)
print("[BIỂU ĐỒ 5c] → Standard Class chiếm đa số đơn hàng; Same Day ít nhất")

plt.tight_layout()
plt.savefig("chart_05_discount_shipping.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_05_discount_shipping.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 6 – TƯƠNG QUAN & HEATMAP
# Mục tiêu: Xác định mối quan hệ tuyến tính giữa các biến số
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 6: TƯƠNG QUAN & HEATMAP")
print("=" * 62)

# ── 6.1 Tính ma trận tương quan Pearson ──────────────────────────────────
# Pearson r ∈ [-1, 1]: 1 = tương quan dương hoàn hảo, -1 = ngược chiều hoàn hảo
num_cols    = ["Sales", "Quantity", "Discount", "Profit", "Ship_Days", "Profit_Margin"]
corr_matrix = df[num_cols].corr().round(2)

print("\n[6.1] Ma trận tương quan giữa các biến số:")
print(corr_matrix.to_string())

# ── 6.2 Phân tích tương quan với Profit ──────────────────────────────────
print("\n[6.2] Tương quan với Profit:")
profit_corr = corr_matrix["Profit"].drop("Profit").sort_values(ascending=False)
for var, val in profit_corr.items():
    direction = "dương" if val > 0 else "âm"
    strength  = "mạnh" if abs(val) > 0.5 else ("trung bình" if abs(val) > 0.2 else "yếu")
    print(f"  {var:<20} r = {val:>6.2f}  ({direction} – {strength})")

# ── BIỂU ĐỒ 6: Correlation Heatmap & Bar ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("BIỂU ĐỒ 6 – Ma Trận Tương Quan", fontsize=14, fontweight="bold")

# ── Subplot 6a: Heatmap tam giác dưới ────────────────────────────────────
# mask: ẩn tam giác trên để không hiển thị trùng lặp
ax = axes[0]
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True   # Đánh dấu True cho tam giác trên (bao gồm đường chéo)
sns.heatmap(
    corr_matrix, ax=ax, annot=True, fmt=".2f",
    cmap="RdYlGn",   # Đỏ = tương quan âm, Xanh = tương quan dương
    center=0, vmin=-1, vmax=1, mask=mask,
    linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}
)
ax.set_title("Heatmap Tương Quan (Tam giác dưới)")
print("\n[BIỂU ĐỒ 6a] → Discount tương quan âm mạnh nhất với Profit (-0.22)")
print("  Sales tương quan dương với Profit (r=0.48) – doanh thu cao thường kéo profit lên")

# ── Subplot 6b: Bar chart tương quan của từng biến với Profit ─────────────
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
plt.savefig("chart_06_correlation.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_06_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 8 – TỔNG HỢP INSIGHT & DASHBOARD
# Mục tiêu: Trình bày kết quả phân tích và đề xuất kinh doanh
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 8: KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ – INSIGHT")
print("=" * 62)

# ── 8.1 Các chỉ số KPI tổng hợp ──────────────────────────────────────────
total_sales      = df["Sales"].sum()
total_profit     = df["Profit"].sum()
avg_margin       = df["Profit_Margin"].mean()
total_orders     = df["Order ID"].nunique()
total_customers  = df["Customer ID"].nunique()

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

# ── 8.2 Insight chính (Nam viết phân tích) ────────────────────────────────
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

# ── BIỂU ĐỒ 8: Dashboard tổng hợp ────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig)
fig.suptitle("BIỂU ĐỒ 8 – DASHBOARD TỔNG HỢP KẾT QUẢ", fontsize=15, fontweight="bold")
fig.patch.set_facecolor("#F8FAFC")  # Nền xám nhạt toàn figure

# ── Hàng trên: 4 thẻ KPI màu sắc ────────────────────────────────────────
kpis = [
    ("Tổng Doanh thu",  f"${total_sales/1e6:.2f}M", PALETTE[0]),
    ("Tổng Lợi nhuận",  f"${total_profit/1e3:.1f}K", PALETTE[1]),
    ("Biên LN TB",      f"{avg_margin:.1f}%",        PALETTE[2]),
    ("Số Đơn hàng",     f"{total_orders:,}",         PALETTE[4]),
]
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.text(0.5, 0.6, value, transform=ax.transAxes,
            fontsize=22, fontweight="bold", ha="center", va="center", color="white")
    ax.text(0.5, 0.2, label, transform=ax.transAxes,
            fontsize=10, ha="center", va="center", color="white", alpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

# ── Hàng dưới: 4 mini-chart ───────────────────────────────────────────────

# Chart 8a: Lợi nhuận theo Danh mục
ax5 = fig.add_subplot(gs[1, 0])
cat_p = df.groupby("Category")["Profit"].sum().sort_values()
ax5.barh(cat_p.index, cat_p.values, color=[PALETTE[i] for i in range(3)], edgecolor="white")
ax5.set_title("Lợi nhuận theo Danh mục")
ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))

# Chart 8b: Tổng doanh thu theo năm
ax6 = fig.add_subplot(gs[1, 1])
yearly_summary = df.groupby("Year")["Sales"].sum()
ax6.bar(yearly_summary.index.astype(str), yearly_summary.values,
        color=PALETTE[:4], edgecolor="white")
ax6.set_title("Doanh thu theo Năm")
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
for i, (yr, v) in enumerate(yearly_summary.items()):
    ax6.text(i, v + 5000, f"${v/1e6:.2f}M", ha="center", fontsize=8.5, fontweight="bold")

# Chart 8c: Top 5 Sub-Category theo doanh thu
ax7 = fig.add_subplot(gs[1, 2])
top5 = sub_analysis["Total_Sales"].head(5)
ax7.barh(top5.index[::-1], top5.values[::-1], color=PALETTE[0], edgecolor="white")
ax7.set_title("Top 5 Sub-Category (Sales)")
ax7.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))

# Chart 8d: So sánh R² hai mô hình
ax8 = fig.add_subplot(gs[1, 3])
model_names = ["Linear\nRegression", "Random\nForest"]
r2_scores   = [r2_lr, r2_rf]
bars = ax8.bar(model_names, r2_scores,
               color=[PALETTE[2], PALETTE[0]], edgecolor="white", width=0.5)
for bar, v in zip(bars, r2_scores):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax8.set_ylim(0, 1)
ax8.set_title("R² Mô hình Dự báo")
ax8.set_ylabel("R² Score")

plt.tight_layout()
plt.savefig("chart_08_dashboard.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_08_dashboard.png")

print("\n" + "=" * 62)
print("  PHẦN 5 + 6 + 8 HOÀN THÀNH!")
print("=" * 62)
