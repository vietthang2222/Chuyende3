import streamlit as st
import pandas as pd

#Hà Quang Hà
def finalize_and_export(df_clean):
    """
    Hàm này bây giờ chỉ nhận 1 tham số duy nhất là df_clean
    vì việc lọc nhiễu đã được thực hiện ngầm ở main.py
    """
    st.markdown("---")
    st.header("💡 Bước 4: Kết luận & Tải dữ liệu sạch")
    
    # Hiển thị thông tin kết luận
    st.success(f"🎯 Tổng số dòng dữ liệu sạch cuối cùng: {len(df_clean)}")
    
    # Hiển thị lại bảng dữ liệu một lần nữa ở cuối để kiểm tra trước khi tải
    st.subheader("📋 Xem lại dữ liệu trước khi xuất file")
    st.dataframe(df_clean.head(10))
    
    st.info("Dựa trên phân tích, dữ liệu cho thấy mối liên hệ mật thiết giữa Doanh số và Lợi nhuận sau khi đã loại bỏ các yếu tố nhiễu.")

    # Tạo nút tải file
    csv_data = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải xuống Dataset sạch cuối cùng (CSV)",
        data=csv_data,
        file_name='Cleaned_Data_Final_Group10.csv',
        mime='text/csv'
    )
