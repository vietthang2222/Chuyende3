import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Đỗ Viết Tuấn
def run_prediction(df, col_sales, col_profit):
    st.header("🤖 Bước 3: Dự báo Lợi nhuận (Linear Regression)")
    if col_sales and col_profit:
        X = df[[col_sales]].values
        y = df[col_profit].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Độ chính xác (R2 Score)", f"{metrics.r2_score(y_test, y_pred):.2f}")
        col_m2.metric("Sai số MAE", f"{metrics.mean_absolute_error(y_test, y_pred):.1f}")
        col_m3.metric("Sai số RMSE", f"{np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.1f}")
        
        st.subheader("Biểu đồ: Giá trị Thực tế vs Dự báo")
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.scatter(X_test, y_test, alpha=0.4, label='Dữ liệu thực tế')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Đường dự báo mô hình')
        plt.legend()
        st.pyplot(fig)