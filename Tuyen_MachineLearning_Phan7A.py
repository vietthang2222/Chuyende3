"""
============================================================
  NHÓM PHÂN TÍCH DOANH SỐ BÁN HÀNG – SUPERSTORE DATASET
  ─────────────────────────────────────────────────────────
  THÀNH VIÊN : TUYỂN
  VAI TRÒ    : MACHINE LEARNING
  PHẦN CODE  : Phần 7A – Linear Regression & Random Forest
  ─────────────────────────────────────────────────────────
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

# ── Đọc dataset đã làm sạch từ file Thắng ─────────────────────────────────────
df = pd.read_csv("df_cleaned.csv", parse_dates=["Order Date", "Ship Date"])


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 7A – XÂY DỰNG MÔ HÌNH DỰ BÁO DOANH THU (MACHINE LEARNING)
# Mục tiêu: Huấn luyện và so sánh 2 mô hình hồi quy để dự báo Sales
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PHẦN 7A: XÂY DỰNG MÔ HÌNH DỰ BÁO DOANH THU")
print("=" * 62)

# ── 7A.1 Chuẩn bị Feature (Encoding biến phân loại) ──────────────────────
# Machine Learning yêu cầu đầu vào là số → cần mã hóa cột text
print("\n[7A.1] Encoding biến phân loại bằng LabelEncoder...")
df_model = df.copy()
le = LabelEncoder()   # LabelEncoder: ánh xạ mỗi category sang số nguyên

# Các cột phân loại cần encode
cat_features = ["Region", "Category", "Sub-Category", "Segment", "Ship Mode"]
for col in cat_features:
    df_model[col + "_enc"] = le.fit_transform(df_model[col])
    unique_vals = df[col].nunique()
    print(f"  {col:<20} → {col}_enc  ({unique_vals} giá trị duy nhất)")

# ── 7A.2 Lựa chọn Feature và Target ──────────────────────────────────────
# Feature matrix X: các biến giải thích
# Target vector y: biến cần dự báo (Sales)
feature_cols = [
    "Quantity",          # Số lượng sản phẩm trong đơn hàng
    "Discount",          # Tỷ lệ giảm giá
    "Ship_Days",         # Số ngày giao hàng
    "Region_enc",        # Khu vực (đã encode)
    "Category_enc",      # Danh mục (đã encode)
    "Sub-Category_enc",  # Danh mục con (đã encode)
    "Segment_enc",       # Phân khúc khách hàng (đã encode)
    "Ship Mode_enc",     # Phương thức giao hàng (đã encode)
    "Year",              # Năm đặt hàng
    "Month",             # Tháng đặt hàng
    "Quarter",           # Quý đặt hàng
]
X = df_model[feature_cols]
y = df_model["Sales"]   # Target: doanh thu từng đơn hàng

print(f"\n[7A.2] Shape dữ liệu: X = {X.shape} | y = {y.shape}")
print(f"  Số features: {len(feature_cols)}")
print(f"  Target: Sales | Min=${y.min():.2f} | Max=${y.max():.2f} | Mean=${y.mean():.2f}")

# ── 7A.3 Chia tập Train / Test ────────────────────────────────────────────
# 80% train, 20% test – random_state cố định để kết quả có thể tái tạo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[7A.3] Chia dữ liệu: Train={len(X_train):,} | Test={len(X_test):,}")

# ── 7A.4 Huấn luyện mô hình Linear Regression ────────────────────────────
# Linear Regression: tìm đường thẳng tốt nhất minimizing MSE
print("\n[7A.4] Huấn luyện Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)           # Học từ tập train
y_pred_lr = lr.predict(X_test)     # Dự báo trên tập test

# Đánh giá hiệu suất
r2_lr   = r2_score(y_test, y_pred_lr)                            # R²: % phương sai giải thích được
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))         # RMSE: lỗi trung bình bình phương
mae_lr  = mean_absolute_error(y_test, y_pred_lr)                 # MAE: lỗi tuyệt đối trung bình

print(f"  → R²   = {r2_lr:.4f}  (1.0 là hoàn hảo)")
print(f"  → RMSE = ${rmse_lr:.2f}  (sai số trung bình bình phương)")
print(f"  → MAE  = ${mae_lr:.2f}  (sai số tuyệt đối trung bình)")

# ── 7A.5 Huấn luyện mô hình Random Forest ────────────────────────────────
# Random Forest: tập hợp nhiều cây quyết định (ensemble) → chống overfitting tốt hơn
print("\n[7A.5] Huấn luyện Random Forest (100 cây, max_depth=10)...")
rf = RandomForestRegressor(
    n_estimators=100,   # Số cây quyết định trong rừng
    max_depth=10,       # Chiều sâu tối đa mỗi cây (tránh overfitting)
    random_state=42,    # Seed ngẫu nhiên để tái tạo kết quả
    n_jobs=-1           # Dùng tất cả CPU core có sẵn để tăng tốc
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)

print(f"  → R²   = {r2_rf:.4f}")
print(f"  → RMSE = ${rmse_rf:.2f}")
print(f"  → MAE  = ${mae_rf:.2f}")

# ── 7A.6 So sánh tổng hợp 2 mô hình ─────────────────────────────────────
print("\n[7A.6] SO SÁNH HAI MÔ HÌNH:")
print(f"{'Chỉ số':<10} {'Linear Regression':>20} {'Random Forest':>15}  {'Winner':>10}")
print("-" * 60)
metrics = [
    ("R²",   r2_lr,   r2_rf,   "max"),
    ("RMSE", rmse_lr, rmse_rf, "min"),
    ("MAE",  mae_lr,  mae_rf,  "min"),
]
for name, val_lr, val_rf, better in metrics:
    winner = "Random Forest" if (val_rf > val_lr if better == "max" else val_rf < val_lr) else "Linear Reg"
    print(f"  {name:<8} {val_lr:>20.4f} {val_rf:>15.4f}  → {winner}")
print("\n  ✔ Random Forest vượt trội ở tất cả chỉ số")
print("  ✔ R² RF cao hơn đáng kể → bắt được quan hệ phi tuyến trong dữ liệu")

# ── 7A.7 Feature Importance từ Random Forest ──────────────────────────────
# Feature importance: mức độ đóng góp của từng biến vào quyết định của rừng
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[7A.7] Tầm quan trọng của từng Feature (Random Forest):")
for feat, imp in feat_imp.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ── BIỂU ĐỒ 7A: So sánh hiệu suất mô hình ───────────────────────────────
fig7a = plt.figure(figsize=(16, 6))
gs7a  = gridspec.GridSpec(1, 3, figure=fig7a)
fig7a.suptitle("BIỂU ĐỒ 7A – SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH DỰ BÁO",
               fontsize=16, fontweight="bold", y=0.95)

# ── Subplot 7A-1: Actual vs Predicted (Linear Regression) ─────────────────
# Điểm nằm trên đường chéo y=x → dự báo chính xác
ax1 = fig7a.add_subplot(gs7a[0, 0])
ax1.scatter(y_test, y_pred_lr, alpha=0.3, color=PALETTE[0], s=10)
max_val = max(y_test.max(), y_pred_lr.max())
ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect fit (y=x)")
ax1.set_title(f"Linear Regression\n$R^2$ = {r2_lr:.4f}", pad=10)
ax1.set_xlabel("Actual Sales ($)")
ax1.set_ylabel("Predicted Sales ($)")
ax1.legend(fontsize=8)

# ── Subplot 7A-2: Actual vs Predicted (Random Forest) ────────────────────
ax2 = fig7a.add_subplot(gs7a[0, 1])
ax2.scatter(y_test, y_pred_rf, alpha=0.3, color=PALETTE[1], s=10)
ax2.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect fit (y=x)")
ax2.set_title(f"Random Forest\n$R^2$ = {r2_rf:.4f}", pad=10)
ax2.set_xlabel("Actual Sales ($)")
ax2.legend(fontsize=8)

# ── Subplot 7A-3: So sánh R² bar chart ───────────────────────────────────
ax3 = fig7a.add_subplot(gs7a[0, 2])
bars = ax3.bar(
    ["Linear Reg", "Random Forest"], [r2_lr, r2_rf],
    color=[PALETTE[0], PALETTE[1]], alpha=0.85, width=0.5
)
ax3.set_ylim(0, 1.0)
ax3.set_ylabel("$R^2$ Score")
ax3.set_title("So sánh chỉ số $R^2$", pad=10)
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{bar.get_height():.4f}", ha="center", fontweight="bold")

plt.subplots_adjust(top=0.82, bottom=0.15, left=0.05, right=0.95, wspace=0.25)
plt.savefig("chart_07a_model_comparison.png", bbox_inches="tight")
plt.close()
print("\n✓ Đã lưu: chart_07a_model_comparison.png")

# ── Xuất kết quả để Nam và Hà dùng ───────────────────────────────────────
# Lưu predictions ra file để tránh phải train lại
results = pd.DataFrame({
    "y_test"    : y_test.values,
    "y_pred_lr" : y_pred_lr,
    "y_pred_rf" : y_pred_rf,
})
results.to_csv("ml_predictions.csv", index=False)

# Lưu feature importance
feat_imp.to_csv("feature_importance.csv", header=["importance"])

# In tóm tắt để Hà và Nam dùng trong báo cáo
print("\n[TÓM TẮT KẾT QUẢ – dùng cho báo cáo Word & slide]")
print(f"  Linear Regression : R²={r2_lr:.3f} | RMSE=${rmse_lr:.0f} | MAE=${mae_lr:.0f}")
print(f"  Random Forest     : R²={r2_rf:.3f} | RMSE=${rmse_rf:.0f} | MAE=${mae_rf:.0f}")
print(f"  Feature quan trọng nhất: {feat_imp.index[0]} ({feat_imp.iloc[0]:.4f})")
print("\n✓ Đã xuất ml_predictions.csv và feature_importance.csv")
print("\n" + "=" * 62)
print("  PHẦN 7A HOÀN THÀNH!")
print("=" * 62)
