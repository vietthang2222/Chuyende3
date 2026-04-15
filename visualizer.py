import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

#Nguyễn Hoài Nam
def show_charts(df, col_sales, col_profit, col_cat, col_reg, num_cols):
    st.header("📊 Bước 2: Phân tích khám phá (EDA)")
    tab1, tab2, tab3 = st.tabs(["📈 Xu hướng", "📦 Hạng mục", "🔗 Tương quan"])
    
    with tab1:
        if col_sales:
            st.subheader("Doanh số theo thời gian (Tháng)")
            fig, ax = plt.subplots(figsize=(10, 4))
            df.groupby('Month_Clean')[col_sales].sum().plot(marker='o', color='royalblue', ax=ax)
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if col_cat and col_sales:
                st.subheader(f"Cơ cấu doanh thu theo {col_cat}")
                fig, ax = plt.subplots()
                df.groupby(col_cat)[col_sales].sum().sort_values(ascending=False).head(10).plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette('viridis'))
                st.pyplot(fig)
        with c2:
            if col_reg and col_profit:
                st.subheader(f"Lợi nhuận theo {col_reg}")
                fig, ax = plt.subplots()
                sns.barplot(data=df, x=col_reg, y=col_profit, palette='magma', ax=ax)
                st.pyplot(fig)

    with tab3:
        st.subheader("Ma trận tương quan giữa các biến số")
        if num_cols:
            fig, ax = plt.subplots()
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)