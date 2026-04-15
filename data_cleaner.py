import pandas as pd
import streamlit as st

# Phạm Minh Tuyển
def clean_data(uploaded_file):
    df = None
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            break 
        except Exception:
            continue
    
    if df is not None:
        with st.expander("🛠️ Bước 1: Tiền xử lý dữ liệu (Data Cleaning)", expanded=True):
            # 1. Xóa khoảng trắng tên cột
            df.columns = df.columns.str.strip()
            
            # 2. Xóa trùng lặp
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            duplicates = initial_rows - len(df)
            
            # 3. Điền giá trị khuyết thiếu
            df.fillna(df.mean(numeric_only=True), inplace=True)
            df.fillna("Unknown", inplace=True)
            
            # --- PHẦN HIỂN THỊ BẢNG (THIẾU CỦA BẠN ĐÂY) ---
            st.success(f"✅ Đã xử lý: {duplicates} dòng trùng lặp, điền khuyết thiếu và lọc nhiễu thành công!")
            st.write(f"Số dòng dữ liệu sạch để phân tích: **{len(df)}**")
            st.dataframe(df.head(10)) # Hiện 10 dòng đầu tiên y như ảnh
            
    return df