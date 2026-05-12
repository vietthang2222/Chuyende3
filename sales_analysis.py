# ─── Imports ─────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import io, base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH TRANG & STYLE
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Superstore Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tùy chỉnh
st.markdown("""
<style>
/* Font chung */
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #0f2440 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label { color: #94a3b8 !important; font-size: 12px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 12px !important; color: #64748b !important; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: #0f172a !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Expander (hộp giải thích) */
.streamlit-expanderHeader {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #1e40af !important;
}

/* Info / warning boxes */
.insight-box {
    padding: 14px 18px;
    border-radius: 8px;
    border-left: 4px solid;
    margin: 8px 0;
    font-size: 13.5px;
    line-height: 1.6;
}
.insight-blue  { background:#eff6ff; border-color:#2563eb; color:#1e3a8a; }
.insight-green { background:#f0fdf4; border-color:#059669; color:#064e3b; }
.insight-amber { background:#fffbeb; border-color:#d97706; color:#78350f; }
.insight-red   { background:#fef2f2; border-color:#dc2626; color:#7f1d1d; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg,#2563eb,#3b82f6);
    color: white !important;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 700;
    margin: 20px 0 12px 0;
    letter-spacing: .02em;
}

/* Chart container */
.chart-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MÀU SẮC & HẰNG SỐ
# ════════════════════════════════════════════════════════════════════════════
C = dict(blue="#2563EB", green="#059669", red="#DC2626",
         amber="#D97706", purple="#7C3AED", teal="#0891B2",
         grey="#6B7280")
PALETTE_PX = [C["blue"], C["green"], C["amber"],
              C["red"],  C["purple"], C["teal"]]
CAT_COLOR   = {"Technology": C["blue"],
               "Furniture":  C["amber"],
               "Office Supplies": C["green"]}


# ════════════════════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ════════════════════════════════════════════════════════════════════════════
def fmt_k(v):
    if abs(v) >= 1e6: return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def insight(text, kind="blue"):
    st.markdown(
        f'<div class="insight-box insight-{kind}">{text}</div>',
        unsafe_allow_html=True)

def section(label):
    st.markdown(
        f'<div class="section-header">📌 {label}</div>',
        unsafe_allow_html=True)

def read_how(lines):
    """Hộp 'Cách đọc biểu đồ' dạng expander."""
    with st.expander("💡 Cách đọc biểu đồ này", expanded=False):
        for ln in lines:
            st.markdown(f"- {ln}")

def plotly_cfg():
    """Config mặc định cho plotly: không toolbar rườm rà."""
    return {"displayModeBar": False}


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — TẢI FILE & ĐIỀU HƯỚNG
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Superstore Analytics")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Tải file CSV / Excel",
        type=["csv","xlsx","xls"],
        help="Chọn file Cleaned_Data_Final.csv hoặc bất kỳ file Superstore nào")

    st.markdown("---")
    st.markdown("### 🧭 Điều hướng")
    PAGES = [
        "🏠 Tổng quan & KPI",
        "🧹 Phần 1 — Data Cleaning",
        "🔍 Phần 2 — EDA Tổng quan",
        "📅 Phần 3 — Thời gian & Khu vực",
        "📦 Phần 4 — Sản phẩm & Danh mục",
        "💸 Phần 5 — Discount & Shipping",
        "🔗 Phần 6 — Tương quan",
        "🤖 Phần 7 — Mô hình dự báo",
        "🎯 Phần 8 — Dashboard & Insights",
    ]
    page = st.radio("", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <small style='color:#94a3b8'>
    <b>Hướng dẫn:</b><br>
    1. Tải file CSV lên<br>
    2. Chọn phần phân tích<br>
    3. Đọc giải thích bên dưới mỗi biểu đồ<br><br>
    <b>Dataset:</b> Superstore 2014–2017<br>
    <b>Tools:</b> Python · Streamlit · Plotly
    </small>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ĐỌC & XỬ LÝ DỮ LIỆU  (cache để không đọc lại mỗi lần click)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_clean(file_bytes, fname):
    """Đọc file và thực hiện toàn bộ Data Cleaning (giữ nguyên logic cũ)."""
    if fname.endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes))

    df = df_raw.copy()

    # ── 1.1 Chuyển kiểu ngày ─────────────────────────────────────────────
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  errors="coerce")

    # ── 1.2 Cột phái sinh ────────────────────────────────────────────────
    df["Year"]      = df["Order Date"].dt.year
    df["Month"]     = df["Order Date"].dt.month
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    df["Quarter"]   = df["Order Date"].dt.quarter
    df["Ship_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # ── 1.3–1.4 Duplicate & outlier ──────────────────────────────────────
    n_before  = len(df)
    df.drop_duplicates(inplace=True)
    n_dup = n_before - len(df)

    neg_sales = (df["Sales"] <= 0).sum()
    neg_qty   = (df["Quantity"] <= 0).sum()

    # ── 1.5 Chuẩn hóa văn bản ────────────────────────────────────────────
    for col in ["Region","Category","Sub-Category","Segment","Ship Mode"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # ── 1.6 Profit Margin ────────────────────────────────────────────────
    df["Profit_Margin"] = (df["Profit"] / df["Sales"] * 100).round(2)

    cleaning_log = {
        "Số dòng gốc":      n_before,
        "Trùng lặp đã xóa": n_dup,
        "Sales ≤ 0":        int(neg_sales),
        "Quantity ≤ 0":     int(neg_qty),
        "Số dòng còn lại":  len(df),
        "NULL còn lại":     int(df.isnull().sum().sum()),
    }
    return df_raw, df, cleaning_log


@st.cache_data
def build_models(_df):
    """Huấn luyện Linear Regression & Random Forest (giữ nguyên logic cũ)."""
    df_m = _df.copy()
    le   = LabelEncoder()
    for col in ["Region","Category","Sub-Category","Segment","Ship Mode"]:
        df_m[col+"_enc"] = le.fit_transform(df_m[col])

    feat_cols = ["Quantity","Discount","Ship_Days",
                 "Region_enc","Category_enc","Sub-Category_enc",
                 "Segment_enc","Ship Mode_enc","Year","Month","Quarter"]
    X = df_m[feat_cols]; y = df_m["Sales"]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)

    lr = LinearRegression().fit(X_tr, y_tr)
    y_lr = lr.predict(X_te)

    rf = RandomForestRegressor(n_estimators=100,max_depth=10,
                               random_state=42,n_jobs=-1).fit(X_tr, y_tr)
    y_rf = rf.predict(X_te)

    feat_imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()

    m_agg = _df.groupby(["Year","Month"])["Sales"].sum().reset_index()
    m_agg["t"] = range(len(m_agg))
    lr_ts = LinearRegression().fit(m_agg[["t"]], m_agg["Sales"])
    fut_t = np.arange(len(m_agg), len(m_agg)+12).reshape(-1,1)
    fut_y = lr_ts.predict(fut_t)

    return {
        "r2_lr":  r2_score(y_te, y_lr),
        "rmse_lr":np.sqrt(mean_squared_error(y_te, y_lr)),
        "r2_rf":  r2_score(y_te, y_rf),
        "rmse_rf":np.sqrt(mean_squared_error(y_te, y_rf)),
        "mae_rf": mean_absolute_error(y_te, y_rf),
        "feat_imp": feat_imp,
        "y_te":   y_te, "y_lr": y_lr, "y_rf": y_rf,
        "m_agg":  m_agg, "fut_t": fut_t.flatten(), "fut_y": fut_y,
    }


# ════════════════════════════════════════════════════════════════════════════
# KIỂM TRA FILE ĐÃ TẢI CHƯA
# ════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px'>
      <div style='font-size:64px'>📂</div>
      <h2 style='color:#1e3a5f;margin:16px 0 8px'>Chào mừng đến với Superstore Analytics</h2>
      <p style='color:#64748b;font-size:16px;max-width:520px;margin:0 auto'>
        Tải file <b>Cleaned_Data_Final.csv</b> lên thanh bên trái để bắt đầu
        phân tích toàn bộ dữ liệu bán hàng với 8 phần đầy đủ.
      </p>
      <div style='margin-top:32px;display:flex;gap:16px;justify-content:center;flex-wrap:wrap'>
        <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px 20px;font-size:13px;color:#1e40af'>🧹 Data Cleaning</div>
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:12px 20px;font-size:13px;color:#166534'>🔍 EDA & Visualization</div>
        <div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 20px;font-size:13px;color:#78350f'>📅 Time Series</div>
        <div style='background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px 20px;font-size:13px;color:#991b1b'>🤖 ML Modeling</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Tải & xử lý ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded is not None:
    file_bytes = uploaded.read()
    df_raw, df, clog = load_and_clean(file_bytes, uploaded.name)
else:
    st.warning("Vui lòng upload file trước!")
    st.stop()


# Pre-compute các biến dùng nhiều
cat_sales    = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
seg_sales    = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
top_sub      = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10)
cat_profit   = df.groupby("Category")[["Sales","Profit"]].sum()
cat_profit["Margin%"] = (cat_profit["Profit"]/cat_profit["Sales"]*100).round(1)

sub_analysis = (df.groupby("Sub-Category")
    .agg(Total_Sales=("Sales","sum"),Total_Profit=("Profit","sum"),
         Avg_Discount=("Discount","mean"),Num_Orders=("Order ID","count"))
    .assign(Margin=lambda x:(x["Total_Profit"]/x["Total_Sales"]*100).round(1))
    .sort_values("Total_Sales", ascending=False))

monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly_sales["ym_str"] = monthly_sales["YearMonth"].astype(str)
yearly      = df.groupby("Year")["Sales"].sum()
monthly_avg = df.groupby("Month")["Sales"].mean()
region_year = df.groupby(["Region","Year"])["Sales"].sum().reset_index()

MONTHS_VI = ["Th.1","Th.2","Th.3","Th.4","Th.5","Th.6",
             "Th.7","Th.8","Th.9","Th.10","Th.11","Th.12"]

total_sales    = df["Sales"].sum()
total_profit   = df["Profit"].sum()
avg_margin     = df["Profit_Margin"].mean()
total_orders   = df["Order ID"].nunique()
total_customers= df["Customer ID"].nunique()
loss_pct       = (df["Profit"]<0).mean()*100


# ════════════════════════════════════════════════════════════════════════════
# TRANG 0 — TỔNG QUAN & KPI
# ════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("📊 Superstore Sales Analytics")
    st.markdown(f"**File:** `{uploaded.name}`  ·  **{df.shape[0]:,} dòng**  ·  **{df.shape[1]} cột**  ·  Giai đoạn {int(df['Year'].min())}–{int(df['Year'].max())}")
    st.markdown("---")

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Tổng Doanh thu",    fmt_k(total_sales),   f"+{(yearly.iloc[-1]/yearly.iloc[-2]-1)*100:.0f}% so với năm trước")
    c2.metric("📈 Tổng Lợi nhuận",    fmt_k(total_profit),  f"Biên {total_profit/total_sales*100:.1f}%")
    c3.metric("🧾 Biên LN Trung bình",f"{avg_margin:.1f}%", "Trên từng đơn")
    c4.metric("📦 Đơn hàng",          f"{total_orders:,}",  f"{total_customers} KH")
    c5.metric("⚠️ Tỷ lệ Đơn Lỗ",     f"{loss_pct:.1f}%",   "Profit < 0", delta_color="inverse")

    st.markdown("---")
    section("Snapshot doanh thu theo năm")

    # Mini chart: yearly trend
    fig = px.bar(x=yearly.index.astype(str), y=yearly.values,
                 labels={"x":"Năm","y":"Doanh thu ($)"},
                 color=yearly.index.astype(str),
                 color_discrete_sequence=["#BFDBFE","#93C5FD","#60A5FA","#2563EB"],
                 text=[fmt_k(v) for v in yearly.values])
    fig.update_traces(textposition="outside", textfont_size=13)
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=20,b=20,l=10,r=10), height=280,
                      yaxis=dict(gridcolor="#F3F4F6"),
                      xaxis=dict(title=""))
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    read_how([
        "Mỗi cột = tổng doanh thu trong năm đó.",
        "Màu đậm dần theo năm — năm 2017 là màu đậm nhất.",
        "Số phía trên cột = giá trị tuyệt đối ($).",
        "**Nhận xét:** Doanh thu tăng đều ~30%/năm suốt 4 năm — xu hướng rất tích cực.",
    ])

    st.markdown("---")
    section("Tóm tắt nhanh theo danh mục & khu vực")
    ca, cb = st.columns(2)

    with ca:
        fig2 = px.pie(values=cat_sales.values, names=cat_sales.index,
                      color=cat_sales.index,
                      color_discrete_map=CAT_COLOR, hole=0.55)
        fig2.update_layout(margin=dict(t=20,b=20,l=10,r=10), height=260,
                           legend=dict(orientation="h",yanchor="bottom",y=-0.2))
        fig2.update_traces(textinfo="label+percent", textfont_size=12)
        st.markdown("**Tỷ trọng doanh thu theo Danh mục**")
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    with cb:
        fig3 = px.pie(values=region_sales.values, names=region_sales.index,
                      color_discrete_sequence=PALETTE_PX, hole=0.55)
        fig3.update_layout(margin=dict(t=20,b=20,l=10,r=10), height=260,
                           legend=dict(orientation="h",yanchor="bottom",y=-0.2))
        fig3.update_traces(textinfo="label+percent", textfont_size=12)
        st.markdown("**Tỷ trọng doanh thu theo Khu vực**")
        st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

    insight("📌 <b>Technology</b> chiếm 47% doanh thu và có biên lợi nhuận cao nhất (16.7%).", "blue")
    insight("📌 <b>Furniture</b> doanh thu cao nhưng biên lợi nhuận chỉ 3.2% — cần xem lại chiến lược.", "amber")
    insight("📌 <b>West + East</b> chiếm 65% doanh thu. Central có biên LN thấp nhất (&lt;3%).", "green")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 1 — DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("🧹 Phần 1 — Data Cleaning")
    st.markdown("Quy trình làm sạch dữ liệu: chuyển kiểu ngày, tạo cột phái sinh, kiểm tra giá trị bất thường và chuẩn hóa văn bản.")
    st.markdown("---")

    section("Kết quả làm sạch")
    cols = st.columns(len(clog))
    icons = ["📄","🗑️","⚠️","⚠️","✅","🔍"]
    colors = ["#2563EB","#DC2626","#D97706","#D97706","#059669","#059669"]
    for col, (k,v), ic, co in zip(cols, clog.items(), icons, colors):
        col.markdown(f"""
        <div style='background:white;border:1px solid #e2e8f0;border-radius:10px;
                    padding:16px;text-align:center;border-top:3px solid {co}'>
            <div style='font-size:22px'>{ic}</div>
            <div style='font-size:22px;font-weight:700;color:{co};margin:4px 0'>{v:,}</div>
            <div style='font-size:11px;color:#64748b;text-transform:uppercase;
                        letter-spacing:.05em'>{k}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    section("Dữ liệu thô (5 dòng đầu)")
    st.dataframe(df_raw.head(), use_container_width=True)

    section("Dữ liệu sau làm sạch (5 dòng đầu)")
    st.dataframe(df.head(), use_container_width=True)

    section("Thống kê mô tả")
    st.dataframe(df[["Sales","Quantity","Discount","Profit","Ship_Days","Profit_Margin"]]
                 .describe().round(2), use_container_width=True)

    st.markdown("---")
    section("Phân phối 3 biến chính")

    ca, cb, cc = st.columns(3)

    # Sales distribution
    with ca:
        fig = px.histogram(df, x="Sales", nbins=55, color_discrete_sequence=[C["blue"]],
                           labels={"Sales":"Giá trị đơn hàng ($)","count":"Số đơn"})
        fig.add_vline(x=df["Sales"].mean(),   line_dash="dash", line_color=C["red"],
                      annotation_text=f"Mean ${df['Sales'].mean():.0f}",
                      annotation_position="top right", annotation_font_color=C["red"])
        fig.add_vline(x=df["Sales"].median(), line_dash="dash", line_color=C["amber"],
                      annotation_text=f"Median ${df['Sales'].median():.0f}",
                      annotation_position="top left", annotation_font_color=C["amber"])
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=30,b=10,l=10,r=10), height=260,
                          title=dict(text="Phân phối Sales", font_size=13, font_color="#0f172a"))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    # Ship_Days
    with cb:
        fig2 = px.histogram(df, x="Ship_Days", nbins=18, color_discrete_sequence=[C["green"]],
                            labels={"Ship_Days":"Số ngày giao","count":"Số đơn"})
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=30,b=10,l=10,r=10), height=260,
                           title=dict(text="Phân phối Ngày giao hàng", font_size=13, font_color="#0f172a"))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    # Discount
    with cc:
        fig3 = px.histogram(df, x="Discount", nbins=18, color_discrete_sequence=[C["amber"]],
                            labels={"Discount":"Mức chiết khấu","count":"Số đơn"})
        fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=30,b=10,l=10,r=10), height=260,
                           title=dict(text="Phân phối Discount", font_size=13, font_color="#0f172a"))
        st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Histogram** = biểu đồ cột thể hiện tần suất — trục X là giá trị, trục Y là số đơn hàng có giá trị đó.",
        "**Sales (trái):** Đường đỏ = Mean, đường cam = Median. Mean > Median → phân phối *lệch phải*: đa số đơn nhỏ nhưng vài đơn lớn kéo trung bình lên.",
        "**Ship Days (giữa):** Hầu hết đơn hàng giao trong 0–7 ngày. Cột cao nhất ở 0 = giao cùng ngày.",
        "**Discount (phải):** Phân phối rời rạc — discount chỉ ở các mức cố định (0%, 10%, 20%...). Cột đầu tiên cao nhất = phần lớn đơn không giảm giá.",
    ])

    insight(f"✅ Sau làm sạch: <b>{df.shape[0]:,} dòng</b> · {clog['Trùng lặp đã xóa']} bản ghi trùng đã xóa · 0 NULL còn lại.", "green")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 2 — EDA TỔNG QUAN
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("🔍 Phần 2 — Phân tích khám phá (EDA)")
    st.markdown("6 góc nhìn tổng quan về doanh thu, lợi nhuận và phân bổ theo nhóm sản phẩm, khu vực, phân khúc khách hàng.")
    st.markdown("---")

    section("Doanh thu theo Danh mục / Khu vực / Phân khúc")
    ca, cb, cc = st.columns(3)

    with ca:
        fig = px.bar(x=cat_sales.index, y=cat_sales.values,
                     color=cat_sales.index, color_discrete_map=CAT_COLOR,
                     labels={"x":"","y":"Doanh thu ($)"},
                     text=[fmt_k(v) for v in cat_sales.values])
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=10,b=10,l=5,r=5), height=280,
                          title="Theo Danh mục", yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    with cb:
        pct_r = (region_sales/region_sales.sum()*100).round(1)
        fig2 = px.bar(x=region_sales.index, y=region_sales.values,
                      color=region_sales.index,
                      color_discrete_sequence=PALETTE_PX,
                      labels={"x":"","y":"Doanh thu ($)"},
                      text=[f"{p:.1f}%" for p in pct_r.values])
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=5,r=5), height=280,
                           title="Theo Khu vực", yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    with cc:
        fig3 = px.bar(x=seg_sales.index, y=seg_sales.values,
                      color=seg_sales.index,
                      color_discrete_sequence=[C["red"],C["purple"],C["teal"]],
                      labels={"x":"","y":"Doanh thu ($)"},
                      text=[fmt_k(v) for v in seg_sales.values])
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=5,r=5), height=280,
                           title="Theo Phân khúc KH", yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Biểu đồ cột đứng:** chiều cao cột = tổng doanh thu. So sánh trực quan giữa các nhóm.",
        "**Số trên đỉnh cột:** giá trị tuyệt đối hoặc % tỷ trọng so với tổng.",
        "**Cột khu vực:** % trên đỉnh = tỷ trọng so với tổng 4 vùng. West 35% = lớn nhất.",
    ])

    st.markdown("---")
    section("Top 10 Sub-Category & Sales vs Profit")
    ca, cb = st.columns([1.1, 1])

    with ca:
        colors_bar = [C["red"] if sub_analysis.loc[idx,"Total_Profit"]<0 else C["blue"]
                      for idx in top_sub.index]
        fig4 = go.Figure(go.Bar(
            x=top_sub.values[::-1], y=top_sub.index[::-1],
            orientation="h",
            marker_color=list(reversed(colors_bar)),
            text=[fmt_k(v) for v in top_sub.values[::-1]],
            textposition="outside"))
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=60), height=320,
                           title="Top 10 Sub-Category theo Doanh thu",
                           xaxis=dict(title="", gridcolor="#F3F4F6"),
                           yaxis=dict(title=""))
        st.plotly_chart(fig4, use_container_width=True, config=plotly_cfg())

    with cb:
        scatter_data = []
        for cat, row in cat_profit.iterrows():
            scatter_data.append(dict(cat=cat,sales=row["Sales"],
                                     profit=row["Profit"], margin=row["Margin%"]))
        sdf = pd.DataFrame(scatter_data)
        sdf["marker_size"] = sdf["margin"].clip(lower=0).fillna(0)
        fig5 = px.scatter(sdf, x="sales", y="profit", size="marker_size",
                          color="cat", color_discrete_map=CAT_COLOR,
                          text="cat", labels={"sales":"Doanh thu ($)","profit":"Lợi nhuận ($)","cat":"Danh mục"},
                          size_max=60)
        fig5.add_hline(y=0, line_dash="dash", line_color=C["red"], opacity=0.6)
        fig5.update_traces(textposition="top center")
        fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=10), height=320,
                           title="Sales vs Profit (cỡ = biên LN)",
                           legend=dict(orientation="h",yanchor="top",y=-0.15),
                           xaxis=dict(gridcolor="#F3F4F6"),
                           yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig5, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Cột nằm ngang (trái):** thanh dài = doanh thu cao. **Màu ĐỎ = sản phẩm đang lỗ** (Profit âm), màu xanh = có lãi.",
        "**Scatter plot (phải):** mỗi chấm = 1 danh mục. Trục X = doanh thu, trục Y = lợi nhuận, cỡ chấm = biên LN (%). Góc phải-trên là lý tưởng nhất.",
        "**Đường đứt đỏ** trong scatter = điểm hòa vốn (Profit = 0). Chấm nào dưới đường = đang lỗ.",
    ])

    insight("⚠️ <b>Tables</b> doanh thu $131K nhưng lỗ ròng $17.7K — cần xem lại giá bán và chính sách chiết khấu ngay.", "red")
    insight("✅ <b>Technology</b>: doanh thu cao nhất VÀ lợi nhuận cao nhất — đây là nhóm sản phẩm chiến lược.", "green")

    st.markdown("---")
    section("Phân phối Biên lợi nhuận (Profit Margin)")
    fig6 = px.histogram(df, x=df["Profit_Margin"].clip(-100,100), nbins=50,
                        color_discrete_sequence=[C["purple"]],
                        labels={"x":"Profit Margin (%)","count":"Số đơn"})
    fig6.add_vline(x=0,  line_dash="dash", line_color=C["red"],   line_width=2,
                   annotation_text="Hòa vốn (0%)", annotation_font_color=C["red"])
    fig6.add_vline(x=avg_margin, line_dash="dash", line_color=C["green"], line_width=2,
                   annotation_text=f"Trung bình ({avg_margin:.1f}%)",
                   annotation_position="top right", annotation_font_color=C["green"])
    fig6.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(t=10,b=10,l=10,r=10), height=280,
                       xaxis=dict(gridcolor="#F3F4F6"), yaxis=dict(gridcolor="#F3F4F6"))
    st.plotly_chart(fig6, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Đường đứt ĐỎ** = điểm hòa vốn (0%). Phần cột bên TRÁI đường đỏ = đơn hàng đang lỗ.",
        "**Đường đứt XANH** = biên lợi nhuận trung bình của toàn bộ đơn hàng.",
        f"**{loss_pct:.1f}%** số đơn hàng có lợi nhuận âm — đây là tỷ lệ đáng lo, nguyên nhân chính là discount quá cao.",
    ])


# ════════════════════════════════════════════════════════════════════════════
# TRANG 3 — THỜI GIAN & KHU VỰC
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("📅 Phần 3 — Phân tích thời gian & khu vực")
    st.markdown("Nhận diện xu hướng tăng trưởng theo năm/tháng, tính mùa vụ và hiệu suất từng khu vực.")
    st.markdown("---")

    section("Xu hướng doanh thu hàng tháng (2014–2017)")
    ms = monthly_sales.copy()
    fig = px.area(ms, x="ym_str", y="Sales",
                  labels={"ym_str":"Tháng","Sales":"Doanh thu ($)"},
                  color_discrete_sequence=[C["blue"]])
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=10,b=10,l=10,r=10), height=300,
                      xaxis=dict(tickangle=45, nticks=16, gridcolor="#F3F4F6"),
                      yaxis=dict(gridcolor="#F3F4F6"))
    fig.update_traces(line_width=1.8)
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Line/Area chart:** trục X = thời gian (từ trái sang phải), trục Y = doanh thu tháng đó.",
        "**Đường đi lên** theo thời gian = doanh thu tăng trưởng đều đặn qua các năm.",
        "**Đỉnh nhọn cuối năm** (T11–T12 mỗi năm) = mùa lễ hội, Black Friday — tính mùa vụ rõ rệt.",
        "**Đáy đầu năm** (T1–T2) = sau mùa lễ, nhu cầu giảm tự nhiên.",
    ])

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        section("Tăng trưởng theo năm (YoY)")
        ydf = yearly.reset_index(); ydf.columns = ["Year","Sales"]
        ydf["growth"] = ydf["Sales"].pct_change()*100
        fig2 = px.bar(ydf, x="Year", y="Sales",
                      color="Sales", color_continuous_scale=["#BFDBFE","#2563EB"],
                      text=[fmt_k(v) for v in ydf["Sales"]],
                      labels={"Year":"Năm","Sales":"Doanh thu ($)"})
        for i in range(1, len(ydf)):
            fig2.add_annotation(
                x=ydf.iloc[i]["Year"], y=ydf.iloc[i]["Sales"]*0.5,
                text=f"+{ydf.iloc[i]['growth']:.0f}%",
                font=dict(color="white",size=13,family="Segoe UI"),
                showarrow=False, bgcolor=C["green"], borderpad=4)
        fig2.update_traces(textposition="outside", textfont_size=12)
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=5,r=5), height=280,
                           yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

        read_how([
            "**Cột màu đậm dần** = năm càng mới càng đậm màu.",
            "**Số màu trắng** bên trong cột = % tăng trưởng so với năm trước (YoY).",
            "**YoY tăng ~30%** liên tiếp 3 năm = tăng trưởng bền vững.",
        ])

    with cb:
        section("Tính mùa vụ — Avg doanh thu theo tháng")
        ma = monthly_avg.reset_index(); ma.columns = ["Month","Avg_Sales"]
        ma["Month_VI"] = MONTHS_VI
        ma["Color"] = ma["Avg_Sales"].apply(
            lambda v: C["blue"] if v==ma["Avg_Sales"].max() else
                      C["red"]  if v==ma["Avg_Sales"].min() else "#93C5FD")
        fig3 = go.Figure(go.Bar(
            x=ma["Month_VI"], y=ma["Avg_Sales"],
            marker_color=ma["Color"],
            text=[f"${v:.0f}" for v in ma["Avg_Sales"]],
            textposition="outside"))
        fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=5,r=5), height=280,
                           xaxis=dict(title=""),
                           yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

        read_how([
            "**Cột XANH ĐẬM** = tháng có doanh thu trung bình cao nhất.",
            "**Cột ĐỎ** = tháng có doanh thu trung bình thấp nhất.",
            "Đây là *trung bình qua tất cả các năm* — loại bỏ tăng trưởng để thấy đúng hành vi mùa vụ.",
            "**Ứng dụng:** nhập nhiều hàng tồn kho trước tháng 10–11, giảm nhập tháng 1–2.",
        ])

    st.markdown("---")
    section("Doanh thu theo Khu vực qua từng năm")
    ry = region_year.copy()
    fig4 = px.bar(ry, x="Year", y="Sales", color="Region", barmode="group",
                  color_discrete_sequence=PALETTE_PX,
                  labels={"Sales":"Doanh thu ($)","Year":"Năm","Region":"Khu vực"})
    fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(t=10,b=10,l=10,r=10), height=300,
                       yaxis=dict(gridcolor="#F3F4F6"),
                       legend=dict(orientation="h",yanchor="bottom",y=1,xanchor="right",x=1))
    st.plotly_chart(fig4, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Grouped bar:** mỗi nhóm cột = 1 năm, mỗi màu = 1 khu vực.",
        "So sánh **cùng màu** qua các nhóm = xem khu vực đó tăng trưởng qua các năm như thế nào.",
        "**West (xanh)** luôn cao nhất và tăng đều đặn — khu vực chiến lược số 1.",
        "**Central** tăng chậm hơn các vùng khác — có thể cần đầu tư marketing thêm.",
    ])


# ════════════════════════════════════════════════════════════════════════════
# TRANG 4 — SẢN PHẨM & DANH MỤC
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("📦 Phần 4 — Sản phẩm & Danh mục")
    st.markdown("Xác định nhóm sản phẩm tạo lãi, nhóm đang thua lỗ và biên lợi nhuận từng sub-category.")
    st.markdown("---")

    section("Biên lợi nhuận theo Sub-Category (Diverging Bar)")
    m_sorted = sub_analysis["Margin"].sort_values()
    fig = go.Figure(go.Bar(
        x=m_sorted.values, y=m_sorted.index, orientation="h",
        marker_color=[C["red"] if v<0 else C["green"] for v in m_sorted.values],
        text=[f"{v:.1f}%" for v in m_sorted.values], textposition="outside"))
    fig.add_vline(x=0, line_color="#374151", line_width=1.2)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=10,b=10,l=10,r=80), height=480,
                      xaxis=dict(title="Profit Margin (%)", gridcolor="#F3F4F6"),
                      yaxis=dict(title=""))
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Diverging bar chart** — cột mở rộng sang 2 hướng từ đường 0%:",
        "Cột sang **PHẢI (xanh lá)** = biên lợi nhuận dương → đang có lãi. *Ví dụ: Labels 43.8% = cứ $1 bán ra lãi 43.8 cent.*",
        "Cột sang **TRÁI (đỏ)** = biên lợi nhuận âm → đang lỗ. *Ví dụ: Tables -13.5% = mỗi $1 bán ra mất 13.5 cent.*",
        "**Đường 0%** = điểm hòa vốn. Chỉ cần xem cột nào sang trái = nhóm cần xử lý khẩn cấp.",
        "**Biên LN quan trọng hơn doanh thu:** Tables doanh thu $131K nhưng biên âm → càng bán càng mất tiền.",
    ])

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        section("Bảng chi tiết Sub-Category")
        display_df = sub_analysis[["Total_Sales","Total_Profit","Margin","Avg_Discount","Num_Orders"]].copy()
        display_df.columns = ["Doanh thu","Lợi nhuận","Biên LN (%)","Avg Discount","Số đơn"]
        display_df["Doanh thu"]   = display_df["Doanh thu"].apply(fmt_k)
        display_df["Lợi nhuận"]   = display_df["Lợi nhuận"].apply(lambda v: f"+{fmt_k(v)}" if v>=0 else f"-{fmt_k(abs(v))}")
        display_df["Biên LN (%)"] = display_df["Biên LN (%)"].apply(lambda v: f"{v:.1f}%")
        display_df["Avg Discount"]= display_df["Avg Discount"].apply(lambda v: f"{v:.1%}")
        st.dataframe(display_df, use_container_width=True, height=380)

    with cb:
        section("Bubble chart: Sales vs Profit")
        bdf = sub_analysis.reset_index()
        bdf["color"] = bdf["Total_Profit"].apply(lambda v: "Lỗ" if v<0 else "Lãi")
        fig2 = px.scatter(bdf, x="Total_Sales", y="Total_Profit",
                          size="Num_Orders", color="color", text="Sub-Category",
                          color_discrete_map={"Lỗ":C["red"],"Lãi":C["blue"]},
                          labels={"Total_Sales":"Doanh thu ($)",
                                  "Total_Profit":"Lợi nhuận ($)",
                                  "Num_Orders":"Số đơn"},
                          size_max=40)
        fig2.add_hline(y=0, line_dash="dash", line_color=C["red"], opacity=0.7)
        fig2.update_traces(textposition="top center", textfont_size=8)
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=10), height=380,
                           xaxis=dict(gridcolor="#F3F4F6"),
                           yaxis=dict(gridcolor="#F3F4F6"),
                           legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Bubble chart:** mỗi bong bóng = 1 Sub-Category. Cỡ bong bóng = số đơn hàng.",
        "**Màu XANH = lãi, ĐỎ = lỗ.** Đường đứt đỏ là ranh giới lãi/lỗ.",
        "Lý tưởng: bong bóng **lớn ở góc phải-trên** = nhiều đơn, doanh thu cao, lợi nhuận tốt.",
        "Tables (đỏ, phải) = doanh thu cao nhưng nằm dưới đường đỏ → đang lỗ.",
    ])

    insight("🚨 <b>3 sub-category lỗ:</b> Tables (-$17.7K), Bookcases (-$3.5K), Supplies (-$1.2K). Nguyên nhân chính: discount trung bình >30%.", "red")
    insight("🏆 <b>Hiệu quả nhất:</b> Labels (43.8%), Copiers (34.4%), Fasteners (31.4%) — biên LN rất cao.", "green")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 5 — DISCOUNT & SHIPPING
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("💸 Phần 5 — Discount & Shipping")
    st.markdown("Đánh giá tác động của chiết khấu lên lợi nhuận và hiệu suất từng phương thức giao hàng.")
    st.markdown("---")

    df["Discount_Bin"] = pd.cut(df["Discount"],
        bins=[-0.01,0,0.1,0.2,0.3,0.5,1.0],
        labels=["0%","1–10%","11–20%","21–30%","31–50%",">50%"])
    disc_profit = df.groupby("Discount_Bin", observed=True)["Profit_Margin"].mean()
    ship_counts = df["Ship Mode"].value_counts()

    ca, cb = st.columns(2)

    with ca:
        section("Biên LN trung bình theo mức Discount")
        fig = go.Figure(go.Bar(
            x=disc_profit.index.astype(str), y=disc_profit.values,
            marker_color=[C["green"] if v>=0 else C["red"] for v in disc_profit.values],
            text=[f"{v:.1f}%" for v in disc_profit.values], textposition="outside"))
        fig.add_hline(y=0, line_color="#374151", line_width=1.2)
        # Vùng nguy hiểm
        fig.add_vrect(x0=2.5, x1=5.5, fillcolor=C["red"], opacity=0.06,
                      annotation_text="Vùng lỗ", annotation_position="top right",
                      annotation_font_color=C["red"])
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=10,b=10,l=5,r=5), height=300,
                          xaxis=dict(title="Mức chiết khấu"),
                          yaxis=dict(title="Avg Profit Margin (%)", gridcolor="#F3F4F6"))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

        read_how([
            "Mỗi cột = **biên LN trung bình** của tất cả đơn có mức discount đó.",
            "**XANH LÁ** = trung bình có lãi. **ĐỎ** = trung bình bị lỗ.",
            "**Đường 0%** = điểm hòa vốn. Cột nào dưới đường = mức discount đó đang gây lỗ.",
            "**Vùng đỏ nhạt** bên phải: tất cả mức discount >20% đều dẫn đến lỗ trung bình.",
            "**Ngưỡng an toàn: ≤ 20% discount.** Discount >50% → lỗ trung bình 114%!",
        ])

    with cb:
        section("Scatter: Discount vs Profit (2,000 đơn ngẫu nhiên)")
        sample = df.sample(min(2000,len(df)), random_state=42)
        fig2 = px.scatter(sample, x="Discount", y="Profit",
                          opacity=0.25, color_discrete_sequence=[C["blue"]],
                          labels={"Discount":"Mức Discount","Profit":"Lợi nhuận ($)"})
        fig2.add_hline(y=0,   line_dash="dash", line_color=C["red"],  line_width=1.5,
                       annotation_text="Profit = 0", annotation_font_color=C["red"])
        fig2.add_vline(x=0.2, line_dash="dash", line_color=C["amber"], line_width=1.5,
                       annotation_text="Discount 20%", annotation_font_color=C["amber"])
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=5,r=5), height=300,
                           xaxis=dict(gridcolor="#F3F4F6"),
                           yaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

        read_how([
            "**Scatter plot:** mỗi chấm = 1 đơn hàng thực tế (2,000 đơn ngẫu nhiên).",
            "**Đường đỏ ngang** = ranh giới lãi/lỗ (Profit = 0). Chấm dưới đường = lỗ.",
            "**Đường cam đứng** = ngưỡng 20% discount.",
            "Phần **bên phải đường cam**: phần lớn chấm rơi xuống dưới đường đỏ → tương quan âm rõ ràng.",
        ])

    st.markdown("---")
    section("Phân bổ Phương thức giao hàng")
    ship_df = ship_counts.reset_index(); ship_df.columns = ["Ship Mode","Count"]
    ship_df["Pct"] = (ship_df["Count"]/ship_df["Count"].sum()*100).round(1)
    fig3 = px.bar(ship_df, x="Ship Mode", y="Count",
                  color="Ship Mode", color_discrete_sequence=PALETTE_PX,
                  text=ship_df.apply(lambda r: f"{r['Count']:,}\n({r['Pct']:.1f}%)",axis=1),
                  labels={"Count":"Số đơn hàng","Ship Mode":""})
    fig3.update_traces(textposition="outside")
    fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(t=10,b=10,l=10,r=10), height=280,
                       yaxis=dict(gridcolor="#F3F4F6"))
    st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Số trên đỉnh cột** = số đơn và % tỷ trọng so với tổng.",
        "**Standard Class** chiếm 61% đơn — khách ưu tiên tiết kiệm chi phí hơn tốc độ.",
        "**Same Day** chỉ 5.5% — cơ hội upsell tốt nếu có chính sách phù hợp.",
    ])

    insight("💡 <b>Khuyến nghị:</b> Thiết lập ngưỡng discount tối đa 20% theo từng danh mục. Cần phê duyệt cấp cao cho mọi mức vượt ngưỡng này.", "amber")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 6 — TƯƠNG QUAN
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("🔗 Phần 6 — Ma trận tương quan")
    st.markdown("Đo lường mức độ tương quan tuyến tính (Pearson r) giữa các biến số — yếu tố nào ảnh hưởng nhiều nhất đến lợi nhuận?")
    st.markdown("---")

    num_cols    = ["Sales","Quantity","Discount","Profit","Ship_Days","Profit_Margin"]
    corr_matrix = df[num_cols].corr().round(2)

    ca, cb = st.columns([1.1, 1])

    with ca:
        section("Heatmap Tương quan Pearson")
        fig = px.imshow(corr_matrix,
                        color_continuous_scale="RdYlGn",
                        zmin=-1, zmax=1,
                        text_auto=".2f",
                        aspect="equal")
        fig.update_coloraxes(colorbar_title="r")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=10,b=10,l=10,r=10), height=380)
        fig.update_traces(textfont_size=12)
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

        read_how([
            "Mỗi ô = hệ số tương quan r ∈ [−1, +1] giữa 2 biến.",
            "**XANH LÁ đậm (r ≈ +1):** khi biến này tăng → biến kia cũng tăng.",
            "**ĐỎ đậm (r ≈ −1):** khi biến này tăng → biến kia giảm.",
            "**Vàng (r ≈ 0):** hai biến không liên quan nhau.",
            "Hàng/cột **Discount–Profit** = màu đỏ → tương quan âm mạnh.",
        ])

    with cb:
        section("Tương quan với Profit")
        profit_corr = corr_matrix["Profit"].drop("Profit").sort_values(ascending=False)
        fig2 = go.Figure(go.Bar(
            x=profit_corr.values, y=profit_corr.index, orientation="h",
            marker_color=[C["green"] if v>0 else C["red"] for v in profit_corr.values],
            text=[f"r = {v:.2f}" for v in profit_corr.values],
            textposition="outside"))
        fig2.add_vline(x=0, line_color="#374151", line_width=1.2)
        # Vùng phân loại
        fig2.add_vrect(x0=0.5, x1=1.0,  fillcolor=C["green"], opacity=0.04,
                       annotation_text="Mạnh (+)", annotation_font_color=C["green"],
                       annotation_position="top right")
        fig2.add_vrect(x0=-1.0, x1=-0.5, fillcolor=C["red"], opacity=0.04,
                       annotation_text="Mạnh (−)", annotation_font_color=C["red"],
                       annotation_position="top left")
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=70), height=380,
                           xaxis=dict(title="Pearson r",range=[-0.65,0.7],gridcolor="#F3F4F6"),
                           yaxis=dict(title=""))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

        read_how([
            "**XANH LÁ:** biến này tăng → Profit có xu hướng tăng.",
            "**ĐỎ:** biến này tăng → Profit có xu hướng giảm.",
            "|r| > 0.5 = **mạnh** · 0.2–0.5 = trung bình · <0.2 = yếu.",
            "**Discount r = −0.46:** kẻ thù số 1 của lợi nhuận.",
            "**Sales r = +0.20:** doanh thu cao *không đảm bảo* lợi nhuận cao.",
        ])

    insight("📌 <b>Discount</b> là yếu tố tác động tiêu cực lớn nhất đến lợi nhuận (r = −0.46). Kiểm soát discount = đòn bẩy nhanh nhất để cải thiện lợi nhuận.", "red")
    insight("📌 <b>Profit_Margin</b> tương quan dương mạnh với Profit (r = +0.50) — đương nhiên vì biên cao → lãi cao.", "blue")


# ════════════════════════════════════════════════════════════════════════════
# TRANG 7 — MÔ HÌNH DỰ BÁO
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.title("🤖 Phần 7 — Mô hình dự báo doanh thu")
    st.markdown("Huấn luyện và so sánh Linear Regression với Random Forest. Dự báo xu hướng 12 tháng tiếp theo.")
    st.markdown("---")

    with st.spinner("Đang huấn luyện mô hình... (lần đầu mất ~10 giây)"):
        m = build_models(df)

    # Metrics
    section("Kết quả đánh giá mô hình")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Linear R²",   f"{m['r2_lr']:.4f}",  "R² → 1 là tốt nhất")
    c2.metric("Linear RMSE", f"${m['rmse_lr']:,.0f}", "Sai số bình phương")
    c3.metric("Forest R²",   f"{m['r2_rf']:.4f}",  f"+{(m['r2_rf']-m['r2_lr']):.4f} so với LR")
    c4.metric("Forest MAE",  f"${m['mae_rf']:,.0f}", "Sai số tuyệt đối TB")

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        section("Actual vs Predicted — Linear Regression")
        scat_df = pd.DataFrame({"Thực tế": m["y_te"].values, "Dự báo": m["y_lr"]})
        fig = px.scatter(scat_df, x="Thực tế", y="Dự báo",
                         opacity=0.35, color_discrete_sequence=[C["blue"]])
        mv = max(scat_df.max().max(), 1)
        fig.add_scatter(x=[0,mv], y=[0,mv], mode="lines",
                        line=dict(color=C["red"],dash="dash",width=1.5),
                        name="Dự báo hoàn hảo")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=10,b=10,l=10,r=10), height=300,
                          xaxis=dict(gridcolor="#F3F4F6"),
                          yaxis=dict(gridcolor="#F3F4F6"),
                          legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    with cb:
        section("Actual vs Predicted — Random Forest")
        scat_df2 = pd.DataFrame({"Thực tế": m["y_te"].values, "Dự báo": m["y_rf"]})
        fig2 = px.scatter(scat_df2, x="Thực tế", y="Dự báo",
                          opacity=0.35, color_discrete_sequence=[C["green"]])
        fig2.add_scatter(x=[0,mv], y=[0,mv], mode="lines",
                         line=dict(color=C["red"],dash="dash",width=1.5),
                         name="Dự báo hoàn hảo")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=10), height=300,
                           xaxis=dict(gridcolor="#F3F4F6"),
                           yaxis=dict(gridcolor="#F3F4F6"),
                           legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    read_how([
        "**Actual vs Predicted scatter:** trục X = giá trị thực tế, trục Y = giá trị mô hình dự báo.",
        "**Đường đỏ đứt** = dự báo hoàn hảo (predicted = actual). Điểm càng **gần đường đỏ** = mô hình càng chính xác.",
        f"**Random Forest (R²={m['r2_rf']:.3f})** tốt hơn hẳn **Linear Regression (R²={m['r2_lr']:.3f})** vì học được quan hệ phi tuyến phức tạp.",
        "**R² > 0.7** là mô hình tốt. **MAE** = sai số trung bình mỗi dự báo tính bằng USD.",
    ])

    st.markdown("---")
    ca, cb = st.columns([1, 1.5])

    with ca:
        section("Feature Importance — Random Forest")
        fi = m["feat_imp"].reset_index(); fi.columns = ["Feature","Importance"]
        fi["Color"] = fi["Importance"].apply(
            lambda v: C["blue"] if v==fi["Importance"].max() else
                      C["red"]  if v==fi["Importance"].min() else "#93C5FD")
        fig3 = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker_color=fi["Color"],
            text=[f"{v:.3f}" for v in fi["Importance"]], textposition="outside"))
        fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=60), height=340,
                           xaxis=dict(title="Importance Score",gridcolor="#F3F4F6"),
                           yaxis=dict(title=""))
        st.plotly_chart(fig3, use_container_width=True, config=plotly_cfg())

        read_how([
            "**Thanh DÀI** = biến đó quan trọng hơn với mô hình khi đưa ra dự báo.",
            "Tổng tất cả thanh = 1.0 (100%).",
            "**XANH ĐẬM** = quan trọng nhất · **ĐỎ** = ít quan trọng nhất.",
            "Biến ít quan trọng có thể loại bỏ mà không giảm nhiều độ chính xác.",
        ])

    with cb:
        section("Dự báo xu hướng doanh thu 2018")
        m_agg = m["m_agg"]
        fut_t = m["fut_t"]; fut_y = m["fut_y"]

        fig4 = go.Figure()
        fig4.add_scatter(x=m_agg["t"], y=m_agg["Sales"], mode="lines+markers",
                         name="Lịch sử (2014–2017)",
                         line=dict(color=C["blue"],width=2), marker=dict(size=4))
        fig4.add_scatter(x=fut_t, y=fut_y, mode="lines+markers",
                         name="Dự báo (2018)",
                         line=dict(color=C["red"],width=2.5,dash="dash"),
                         marker=dict(size=5,symbol="square"))
        fig4.add_scatter(x=np.concatenate([fut_t, fut_t[::-1]]),
                         y=np.concatenate([fut_y*0.85, (fut_y*1.15)[::-1]]),
                         fill="toself", fillcolor=f"rgba(220,38,38,0.08)",
                         line=dict(color="rgba(0,0,0,0)"),
                         name="Vùng tin cậy ±15%")
        fig4.add_vline(x=47.5, line_dash="dot", line_color=C["grey"], opacity=0.6)
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=10), height=340,
                           xaxis=dict(title="Thứ tự tháng (0=T1/2014)",gridcolor="#F3F4F6"),
                           yaxis=dict(title="Doanh thu ($)",gridcolor="#F3F4F6"),
                           legend=dict(orientation="h",yanchor="bottom",y=1,xanchor="right",x=1))
        st.plotly_chart(fig4, use_container_width=True, config=plotly_cfg())

        read_how([
            "**Đường XANH** = doanh thu thực tế đã biết (2014–2017).",
            "**Đường ĐỎ đứt** = dự báo 12 tháng tiếp theo (2018) dựa trên xu hướng tuyến tính.",
            "**Vùng tô nhạt** = khoảng bất định ±15% — thực tế có thể cao hoặc thấp hơn dự báo.",
            "**Đường dọc** = ranh giới giữa lịch sử và dự báo.",
            "Mô hình tuyến tính đơn giản → chưa nắm bắt được mùa vụ, chỉ dùng ước lượng tổng thể.",
        ])


# ════════════════════════════════════════════════════════════════════════════
# TRANG 8 — DASHBOARD & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[8]:
    st.title("🎯 Phần 8 — Dashboard tổng hợp & Insights")
    st.markdown("Tổng kết toàn bộ kết quả phân tích kèm khuyến nghị hành động cụ thể.")
    st.markdown("---")

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Tổng Doanh thu",   fmt_k(total_sales))
    c2.metric("Tổng Lợi nhuận",   fmt_k(total_profit))
    c3.metric("Biên LN TB",       f"{avg_margin:.1f}%")
    c4.metric("Tổng Đơn hàng",    f"{total_orders:,}")
    c5.metric("Tỷ lệ Đơn Lỗ",     f"{loss_pct:.1f}%", delta_color="inverse")

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        section("Lợi nhuận theo Danh mục")
        cat_p = df.groupby("Category")["Profit"].sum().reset_index()
        fig = px.bar(cat_p, y="Category", x="Profit", orientation="h",
                     color="Category", color_discrete_map=CAT_COLOR,
                     text=cat_p["Profit"].apply(fmt_k),
                     labels={"Profit":"Lợi nhuận ($)","Category":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(t=10,b=10,l=10,r=60), height=240,
                          xaxis=dict(gridcolor="#F3F4F6"))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    with cb:
        section("Top 5 Sub-Category theo Doanh thu")
        top5 = sub_analysis.head(5).reset_index()
        top5["color"] = top5["Total_Profit"].apply(lambda v: "Lỗ" if v<0 else "Lãi")
        fig2 = px.bar(top5, y="Sub-Category", x="Total_Sales", orientation="h",
                      color="color", color_discrete_map={"Lỗ":C["red"],"Lãi":C["blue"]},
                      text=top5["Total_Sales"].apply(fmt_k),
                      labels={"Total_Sales":"Doanh thu ($)","Sub-Category":""})
        fig2.update_traces(textposition="outside")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=10,l=10,r=60), height=240,
                           xaxis=dict(gridcolor="#F3F4F6"),
                           legend=dict(orientation="h",y=-0.25))
        st.plotly_chart(fig2, use_container_width=True, config=plotly_cfg())

    st.markdown("---")
    section("7 Insights chính & Khuyến nghị hành động")

    insights_data = [
        ("blue",  "1 · Tăng trưởng bền vững",
         "Doanh thu tăng đều <b>~30–37%/năm</b> trong 4 năm liên tiếp. Năm 2017 đạt $362K, gấp 2.3 lần 2014. Xu hướng tích cực và ổn định."),
        ("green", "2 · Khu vực & Mùa vụ",
         "<b>West</b> dẫn đầu doanh thu (35%). Tháng 11–12 cao hơn trung bình 60–120% — cần chuẩn bị hàng tồn kho từ tháng 9 để đáp ứng nhu cầu Q4."),
        ("blue",  "3 · Danh mục chiến lược",
         "<b>Technology</b>: doanh thu cao nhất ($384K) VÀ lợi nhuận cao nhất (biên 16.7%). Là danh mục cần ưu tiên đầu tư và mở rộng."),
        ("amber", "4 · Furniture — nghịch lý",
         "<b>Furniture</b> doanh thu $287K (hạng 2) nhưng biên lợi nhuận chỉ <b>3.2%</b> — gần như hòa vốn. Nguyên nhân: discount cao + chi phí vận chuyển lớn."),
        ("red",   "5 · Sản phẩm lỗ cần xử lý",
         "<b>Tables lỗ $17.7K, Bookcases lỗ $3.5K, Supplies lỗ $1.2K</b>. Discount trung bình >30% là nguyên nhân chính. Cần điều chỉnh giá hoặc dừng chiết khấu sâu."),
        ("red",   "6 · Chính sách Discount",
         "Discount >20% → biên LN âm trung bình. <b>18.3% đơn hàng đang lỗ.</b> Cần thiết lập ngưỡng tối đa 20% và yêu cầu phê duyệt cấp cao cho mọi mức vượt ngưỡng."),
        ("green", "7 · Mô hình dự báo",
         f"Random Forest đạt <b>R²={build_models(df)['r2_rf']:.3f}</b>, MAE=${build_models(df)['mae_rf']:,.0f} — mô hình đáng tin cậy. Dự báo 2018 cho thấy xu hướng tiếp tục tăng trưởng tích cực."),
    ]
    for kind, title, body in insights_data:
        insight(f"<b>{title}</b><br>{body}", kind)
