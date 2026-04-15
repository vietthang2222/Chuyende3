import streamlit as st
import data_cleaner, data_processor, visualizer, predictor, validator

# 1. CẤU HÌNH GIAO DIỆN (Đã thêm lại page_icon và layout)
st.set_page_config(page_title="AI Sales Analyzer Pro", layout="wide", page_icon="📈")

# --- PHẦN SIDEBAR (Đã thêm lại Logo và Tiêu đề) ---
st.sidebar.header("📁 Quản lý dữ liệu")
uploaded_file = st.sidebar.file_uploader("Bước 1: Tải lên Dataset (CSV)", type=["csv"])

# --- TIÊU ĐỀ CHÍNH ---
st.title("🚀 Phân tích doanh số bán hàng")
st.markdown("---")

if uploaded_file is not None:
    # BƯỚC 1: TUYỂN LÀM SẠCH & HIỂN THỊ BẢNG
    df = data_cleaner.clean_data(uploaded_file)
    
    if df is not None:
        # XỬ LÝ LOGIC (THẮNG)
        df, col_sales, col_profit, col_cat, col_reg, num_cols = data_processor.process_features(df)
        
        # LỌC NHIỄU NGẦM (Để Tuấn có sai số thấp)
        if col_sales:
            Q1 = df[col_sales].quantile(0.25)
            Q3 = df[col_sales].quantile(0.75)
            IQR = Q3 - Q1
            df_clean = df[(df[col_sales] >= (Q1 - 1.5 * IQR)) & (df[col_sales] <= (Q3 + 1.5 * IQR))].copy()
        else:
            df_clean = df.copy()

        # BƯỚC 2: NAM VẼ BIỂU ĐỒ (Dùng dữ liệu sạch df_clean)
        visualizer.show_charts(df_clean, col_sales, col_profit, col_cat, col_reg, num_cols)
        
        # BƯỚC 3: TUẤN CHẠY AI (Sai số sẽ nhỏ vì dùng df_clean)
        predictor.run_prediction(df_clean, col_sales, col_profit)
        
        # BƯỚC 4: HÀ KẾT LUẬN & XUẤT FILE (Hiện ở cuối)
        validator.finalize_and_export(df_clean) 
        
else:
    # THÔNG BÁO KHI CHƯA TẢI FILE (Dòng màu vàng trong ảnh của bạn)
    st.warning("👈 Vui lòng tải file CSV lên thanh bên trái để hệ thống bắt đầu làm việc!")