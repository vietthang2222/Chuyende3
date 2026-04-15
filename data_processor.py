import pandas as pd
import numpy as np

#Bùi Đình Việt Thắng
def process_features(df):
    # Tự động nhận diện loại cột
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'ngày' in c.lower()]

    # Gán cột thông minh dựa trên từ khóa gốc
    col_sales = next((c for c in num_cols if 'sales' in c.lower() or 'doanh' in c.lower()), num_cols[0] if num_cols else None)
    col_profit = next((c for c in num_cols if 'profit' in c.lower() or 'lợi' in c.lower()), num_cols[1] if len(num_cols) > 1 else None)
    col_cat = next((c for c in cat_cols if 'category' in c.lower() or 'loại' in c.lower() or 'danh mục' in c.lower()), cat_cols[0] if cat_cols else None)
    col_reg = next((c for c in cat_cols if 'region' in c.lower() or 'vùng' in c.lower() or 'khu' in c.lower()), cat_cols[1] if len(cat_cols) > 1 else None)

    # Xử lý ngày tháng
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df.dropna(subset=[date_cols[0]], inplace=True)
        df['Month_Clean'] = df[date_cols[0]].dt.month
    else:
        df['Month_Clean'] = np.random.randint(1, 13, size=len(df))
        
    return df, col_sales, col_profit, col_cat, col_reg, num_cols