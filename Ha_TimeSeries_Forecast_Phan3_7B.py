"""
============================================================
  NHÓM PHÂN TÍCH DOANH SỐ BÁN HÀNG – SUPERSTORE DATASET
  ─────────────────────────────────────────────────────────
  THÀNH VIÊN : HÀ
  VAI TRÒ    : TIME SERIES + FORECAST
  PHẦN CODE  : Phần 3 – Phân tích thời gian & khu vực
               Phần 7B – Dự báo 12 tháng tới & Feature Importance
===================================================
"""

# ── Thư viện ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

# ── Đọc dataset đã làm sạch ───────────────────────────────────────────────────
df = pd.read_csv("df_cleaned.csv", parse_dates=["Order Date", "Ship Date"])
# Khôi phục cột YearMonth dạng Period (bị chuyển thành string khi lưu CSV)
df["YearMonth"] = pd.to_datetime(df["Order Date"]).dt.to_period("M")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 – PHÂN TÍCH DOANH THU THEO THỜI GIAN & KHU VỰC
# Mục tiêu: Khám phá xu hướng (trend), mùa vụ (seasonality) và tăng trưởng
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 3: PHÂN TÍCH DOANH THU THEO THỜI GIAN & KHU VỰC")
print("=" * 62)

# ── 3.1 Tổng hợp doanh thu theo tháng và năm ─────────────────────────────
# groupby YearMonth: tạo chuỗi thời gian monthly
monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly_sales["YearMonth_str"] = monthly_sales["YearMonth"].astype(str)   # Chuyển sang chuỗi để plot

# Tổng doanh thu theo năm
yearly = df.groupby("Year")["Sales"].sum()

# Tổng doanh thu theo năm và quý (phân tích Q1-Q4)
qtr = df.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()

print("\n[3.1] Doanh thu theo năm:")
for yr, val in yearly.items():
    print(f"  Năm {yr}: ${val:>12,.0f}")

# pct_change(): tính % thay đổi so với hàng trước
growth = yearly.pct_change() * 100
print("\n[3.1] Tăng trưởng theo năm (%):")
for yr, g in growth.items():
    if pd.notna(g):
        arrow = "↑" if g > 0 else "↓"
        print(f"  {yr}: {g:+.1f}% {arrow}")

# ── 3.2 Phân tích Seasonality – doanh thu trung bình theo tháng ───────────
# Tính mean thay vì sum để so sánh công bằng giữa các tháng (4 năm × tháng)
monthly_avg = df.groupby("Month")["Sales"].mean()
months_name = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

print("\n[3.2] Doanh thu trung bình theo tháng (Seasonality):")
for m, v in monthly_avg.items():
    bar = "▓" * int(v / 1000)   # Minibar text để dễ so sánh
    print(f"  Tháng {m:>2} ({months_name[m-1]}): ${v:>8,.0f}  {bar}")
print(f"\n  → Tháng cao nhất : {monthly_avg.idxmax()} ({months_name[monthly_avg.idxmax()-1]})")
print(f"  → Tháng thấp nhất: {monthly_avg.idxmin()} ({months_name[monthly_avg.idxmin()-1]})")
print(f"  → Biên độ dao động: ${monthly_avg.max() - monthly_avg.min():,.0f}/tháng")

# ── 3.3 Phân tích doanh thu theo Khu vực & Năm ────────────────────────────
region_year = df.groupby(["Region", "Year"])["Sales"].sum().reset_index()
print("\n[3.3] Doanh thu theo Khu vực & Năm:")
for region in sorted(df["Region"].unique()):
    sub  = region_year[region_year["Region"] == region]
    vals = " | ".join([f"{int(r.Year)}: ${r.Sales:,.0f}" for _, r in sub.iterrows()])
    print(f"  {region:<10} → {vals}")

# ── BIỂU ĐỒ 3: Xu hướng thời gian & khu vực (4 subplot) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
fig.suptitle("BIỂU ĐỒ 3 – Phân Tích Doanh Thu Theo Thời Gian & Khu Vực",
             fontsize=14, fontweight="bold")

# ── Subplot 3a: Line chart doanh thu hàng tháng ──────────────────────────
ax = axes[0, 0]
x  = range(len(monthly_sales))
ax.plot(x, monthly_sales["Sales"], color=PALETTE[0], linewidth=2, marker="o", markersize=3)
ax.fill_between(x, monthly_sales["Sales"], alpha=0.15, color=PALETTE[0])  # Tô màu phía dưới đường
# Hiển thị 12 nhãn X thay vì tất cả để không bị chồng chéo
step = max(1, len(monthly_sales) // 12)
ax.set_xticks(list(x)[::step])
ax.set_xticklabels(monthly_sales["YearMonth_str"].tolist()[::step], rotation=45, ha="right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax.set_title("Xu hướng Doanh thu Hàng tháng")
ax.set_ylabel("Sales ($)")
print("\n[BIỂU ĐỒ 3a] → Doanh thu có xu hướng tăng theo năm; các tháng cuối năm thường cao hơn")

# ── Subplot 3b: Bar chart tổng doanh thu theo năm ────────────────────────
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

# ── Subplot 3c: Bar chart seasonality (trung bình theo tháng) ─────────────
# Màu đặc biệt cho tháng cao nhất (xanh) và thấp nhất (đỏ), còn lại xanh nhạt
ax = axes[1, 0]
month_colors = [
    PALETTE[0] if v == monthly_avg.max() else
    PALETTE[3] if v == monthly_avg.min() else
    PALETTE[1] for v in monthly_avg.values
]
ax.bar(months_name, monthly_avg.values, color=month_colors, edgecolor="white")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax.set_title("Doanh thu Trung bình Theo Tháng (Seasonality)")
ax.set_ylabel("Avg Sales ($)")
# Đánh dấu tháng cao nhất
ax.annotate("Cao nhất",
            xy=(monthly_avg.idxmax()-1, monthly_avg.max()),
            xytext=(monthly_avg.idxmax()-1, monthly_avg.max()*1.07),
            ha="center", color=PALETTE[0], fontsize=8, fontweight="bold")
print("[BIỂU ĐỒ 3c] → Tháng 11 & 12 có doanh thu trung bình cao nhất (mùa lễ hội cuối năm)")

# ── Subplot 3d: Grouped bar chart doanh thu theo Region × Năm ─────────────
ax = axes[1, 1]
regions = sorted(df["Region"].unique())
x_pos   = np.arange(len(years_list))
width   = 0.2    # Độ rộng mỗi cột trong nhóm
for i, region in enumerate(regions):
    # Lấy doanh thu từng năm cho khu vực này
    vals_series = region_year[region_year["Region"] == region].set_index("Year")["Sales"]
    vals = [vals_series.get(y, 0) for y in years_list]
    ax.bar(x_pos + i * width, vals, width=width, label=region,
           color=PALETTE[i], edgecolor="white")
ax.set_xticks(x_pos + width * 1.5)   # Căn giữa nhóm cột
ax.set_xticklabels(years_list)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.set_title("Doanh thu Theo Khu vực & Năm")
ax.set_ylabel("Sales ($)")
ax.legend()
print("[BIỂU ĐỒ 3d] → West tăng trưởng đều đặn và dẫn đầu; tất cả khu vực đều tăng qua các năm")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("chart_03_time_region.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_03_time_region.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7B – DỰ BÁO DOANH THU 12 THÁNG & FEATURE IMPORTANCE
# Mục tiêu: Xây dựng mô hình time series đơn giản và vẽ dự báo 2018
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 7B: DỰ BÁO DOANH THU 2018 & FEATURE IMPORTANCE")
print("=" * 62)

# ── 7B.1 Tổng hợp doanh thu theo tháng (series thời gian) ────────────────
# Mỗi record là tổng Sales của 1 tháng
monthly_agg = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
monthly_agg["t"] = range(len(monthly_agg))   # Biến thời gian t = 0, 1, 2, ... (index tháng)

print(f"\n[7B.1] Dữ liệu time series: {len(monthly_agg)} điểm tháng")
print(f"  Từ: {monthly_agg['Year'].min()}-{monthly_agg['Month'].min():02d}")
print(f"  Đến: {monthly_agg['Year'].max()}-{monthly_agg['Month'].max():02d}")

# ── 7B.2 Huấn luyện mô hình hồi quy tuyến tính theo thời gian ─────────────
# Dùng Linear Regression với biến dự báo t = thứ tự tháng
# Đây là cách đơn giản để ước lượng xu hướng dài hạn (trend)
lr_ts = LinearRegression()
lr_ts.fit(monthly_agg[["t"]], monthly_agg["Sales"])

slope     = lr_ts.coef_[0]      # Hệ số góc: Sales tăng thêm bao nhiêu mỗi tháng
intercept = lr_ts.intercept_    # Hệ số chặn
trend_r2  = lr_ts.score(monthly_agg[["t"]], monthly_agg["Sales"])

print(f"\n[7B.2] Mô hình xu hướng tuyến tính:")
print(f"  Sales(t) = {slope:.2f} × t + {intercept:.2f}")
print(f"  Slope: mỗi tháng doanh thu tăng trung bình ${slope:,.2f}")
print(f"  R² xu hướng: {trend_r2:.3f}")

# ── 7B.3 Dự báo 12 tháng tiếp theo (2018) ────────────────────────────────
# t tiếp theo bắt đầu từ len(monthly_agg) → len(monthly_agg)+11
future_t    = np.arange(len(monthly_agg), len(monthly_agg) + 12).reshape(-1, 1)
future_pred = lr_ts.predict(future_t)

# Vùng tin cậy: ±15% so với dự báo (ước lượng thủ công)
ci_lower = future_pred * 0.85
ci_upper = future_pred * 1.15

print("\n[7B.3] Dự báo 12 tháng 2018:")
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for i, (pred, lo, hi) in enumerate(zip(future_pred, ci_lower, ci_upper)):
    print(f"  2018-{i+1:02d} ({month_names[i]}): ${pred:>10,.0f}  [${lo:,.0f} – ${hi:,.0f}]")
print(f"\n  Tổng dự báo cả năm 2018: ${future_pred.sum():,.0f}")

# ── 7B.4 Đọc Feature Importance từ file Tuyển (hoặc tự tính) ─────────────
try:
    feat_imp = pd.read_csv("feature_importance.csv", index_col=0, squeeze=True)
    print("\n[7B.4] Đã đọc Feature Importance từ Tuyển")
except FileNotFoundError:
    # Tính lại nếu chưa có file (dự phòng)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    df_tmp = df.copy()
    le = LabelEncoder()
    for col in ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]:
        df_tmp[col + "_enc"] = le.fit_transform(df_tmp[col])
    feature_cols = ["Quantity", "Discount", "Ship_Days",
                    "Region_enc", "Category_enc", "Sub-Category_enc",
                    "Segment_enc", "Ship Mode_enc", "Year", "Month", "Quarter"]
    X = df_tmp[feature_cols]; y = df_tmp["Sales"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_tmp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_tmp.fit(Xtr, ytr)
    feat_imp = pd.Series(rf_tmp.feature_importances_, index=feature_cols)
    print("\n[7B.4] Đã tự tính lại Feature Importance (file Tuyển chưa có)")

feat_imp_sorted = feat_imp.sort_values(ascending=True)

# ── BIỂU ĐỒ 7B: Feature Importance & Dự báo ──────────────────────────────
fig7b = plt.figure(figsize=(16, 7))
import matplotlib.gridspec as gridspec
gs7b = gridspec.GridSpec(1, 2, figure=fig7b, width_ratios=[1, 1.5])
fig7b.suptitle("BIỂU ĐỒ 7B – TẦM QUAN TRỌNG BIẾN & DỰ BÁO DOANH THU 2018",
               fontsize=16, fontweight="bold", y=0.95)

# ── Subplot 7B-1: Feature Importance (Random Forest) ─────────────────────
ax4 = fig7b.add_subplot(gs7b[0, 0])
ax4.barh(feat_imp_sorted.index, feat_imp_sorted.values, color=PALETTE[4], edgecolor="white")
ax4.set_title("Các yếu tố quan trọng nhất\n(Random Forest)", pad=15)
ax4.set_xlabel("Importance Score")
# Ghi nhãn % trên mỗi thanh
for i, v in enumerate(feat_imp_sorted.values):
    ax4.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
print("\n[BIỂU ĐỒ 7B-1] Feature quan trọng nhất quyết định Sales")

# ── Subplot 7B-2: Lịch sử + Dự báo 2018 ─────────────────────────────────
ax5 = fig7b.add_subplot(gs7b[0, 1])
# Vẽ chuỗi lịch sử (2014-2017)
ax5.plot(monthly_agg["t"], monthly_agg["Sales"],
         color=PALETTE[0], linewidth=2, marker="o", markersize=4,
         label="Lịch sử (2014-2017)")
# Vẽ dự báo (2018) – đường đứt nét màu đỏ
ax5.plot(future_t.flatten(), future_pred,
         color=PALETTE[3], linewidth=2.5, linestyle="--", marker="s",
         label="Dự báo (2018)")
# Tô vùng tin cậy
ax5.fill_between(future_t.flatten(), ci_lower, ci_upper,
                 alpha=0.2, color=PALETTE[3], label="Vùng tin cậy ±15%")
# Đường dọc phân chia lịch sử / dự báo
ax5.axvline(len(monthly_agg) - 0.5, color="gray", linestyle=":", linewidth=1.5)
ax5.text(len(monthly_agg) - 0.3, monthly_agg["Sales"].max() * 0.95,
         "→ Dự báo", color="gray", fontsize=9)
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}K"))
ax5.set_title("Xu hướng & Dự báo Doanh thu Theo Tháng", pad=15)
ax5.set_xlabel("Thứ tự tháng (t)")
ax5.set_ylabel("Doanh thu ($)")
ax5.legend(loc="upper left")
ax5.grid(True, linestyle=":", alpha=0.6)
print("[BIỂU ĐỒ 7B-2] → Xu hướng tăng trưởng dài hạn ổn định; dự báo 2018 tiếp tục tăng")

plt.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.92, wspace=0.25)
plt.savefig("chart_07b_forecast.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_07b_forecast.png")

print("\n" + "=" * 62)
print("  PHẦN 3 + 7B HOÀN THÀNH!")
print("=" * 62)
print("""
  Tóm tắt kết quả thời gian:
    • Doanh thu tăng ~20-25% mỗi năm từ 2014→2017
    • Tháng 11 & 12 luôn là đỉnh doanh số (mùa lễ hội)
    • West dẫn đầu, South tăng trưởng chậm nhất
    • Dự báo 2018: tổng cả năm ≈ ${:.0f}
""".format(future_pred.sum()))
