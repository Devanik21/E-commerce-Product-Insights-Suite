# 🛒 E-commerce Product Insights Suite

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="🛒 E-commerce Insights Suite", layout="wide")
st.title("🛒 E-commerce Product Insights Suite")
st.write("Exploring insights from the preloaded Fashion Dataset. Analyze segments, churn, sales trends, and more!")

# Load preloaded data
# Ensure 'FashionDataset.csv' is in the same directory as app.py or provide a full path.
try:
    df = pd.read_csv("FashionDataset.csv")
    st.subheader("🧾 Data Preview (FashionDataset.csv)")
    st.write(df.head())

    st.info("""
    **Note:** The analysis modules below (RFM, Churn, Forecasting, etc.) are designed for datasets with specific column names 
    (e.g., `CustomerID`, `OrderDate`, `TotalAmount`, `ProductID`, `IsReturned`, `Group`, `Conversion`). 
    The preloaded `FashionDataset.csv` may have different column names or may not contain all required columns.
    As a result, some or all analysis features may not function as expected or display any results.
    You may need to adapt the dataset or the analysis code if you wish to use these features with the current data.
    """)

    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    st.sidebar.header("🧠 Analysis Options")

    # RFM Segmentation
    if st.sidebar.checkbox("🧩 RFM Segmentation"):
        st.subheader("🧩 RFM Segmentation")
        if 'CustomerID' in df.columns and 'OrderDate' in df.columns and 'TotalAmount' in df.columns:
            snapshot_date = df['OrderDate'].max() + pd.Timedelta(days=1)
            rfm = df.groupby('CustomerID').agg({
                'OrderDate': lambda x: (snapshot_date - x.max()).days,
                'CustomerID': 'count',
                'TotalAmount': 'sum'
            })
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            st.write(rfm.head())

            # KMeans for RFM Clusters
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=4, random_state=1)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
            st.write(rfm.groupby('Cluster').mean())

            # Plot
            sns.pairplot(rfm.reset_index(), hue='Cluster')
            st.pyplot(plt.gcf())

    # Customer Churn Detection
    if st.sidebar.checkbox("📉 Customer Churn Detection"):
        st.subheader("📉 Customer Churn Detection")
        if 'CustomerID' in df.columns and 'OrderDate' in df.columns:
            recent_orders = df.groupby('CustomerID')['OrderDate'].max()
            churn_threshold = st.slider("Days since last order to be considered churned", 30, 180, 90)
            churned = recent_orders < (df['OrderDate'].max() - pd.Timedelta(days=churn_threshold))
            churn_summary = churned.value_counts().rename(index={True: 'Churned', False: 'Active'})
            st.bar_chart(churn_summary)

    # Sales Forecasting
    if st.sidebar.checkbox("📈 Sales Forecasting"):
        st.subheader("📈 Sales Forecasting (Simple Trend)")
        if 'OrderDate' in df.columns and 'TotalAmount' in df.columns:
            df_sorted = df.sort_values('OrderDate')
            df_grouped = df_sorted.groupby('OrderDate')['TotalAmount'].sum().reset_index()
            df_grouped['Days'] = (df_grouped['OrderDate'] - df_grouped['OrderDate'].min()).dt.days
            model = LinearRegression()
            model.fit(df_grouped[['Days']], df_grouped['TotalAmount'])
            future_days = np.arange(df_grouped['Days'].max() + 1, df_grouped['Days'].max() + 31).reshape(-1, 1)
            future_sales = model.predict(future_days)
            future_dates = [df_grouped['OrderDate'].max() + pd.Timedelta(days=i) for i in range(1, 31)]
            forecast_df = pd.DataFrame({"OrderDate": future_dates, "ForecastedSales": future_sales})
            st.line_chart(pd.concat([df_grouped[['OrderDate', 'TotalAmount']].rename(columns={"TotalAmount": "ForecastedSales"}), forecast_df]))

    # Return Analysis
    if st.sidebar.checkbox("↩️ Return Analysis"):
        st.subheader("↩️ Return Rate by Product")
        if 'ProductID' in df.columns and 'IsReturned' in df.columns:
            return_rate = df.groupby('ProductID')['IsReturned'].mean()
            st.bar_chart(return_rate.sort_values(ascending=False).head(10))

    # A/B Test Visual
    if st.sidebar.checkbox("🧪 A/B Test Summary"):
        st.subheader("🧪 A/B Test Summary for Pricing")
        if 'Group' in df.columns and 'Conversion' in df.columns:
            ab_summary = df.groupby('Group')['Conversion'].agg(['count', 'mean'])
            st.write(ab_summary)
            sns.barplot(data=df, x='Group', y='Conversion')
            st.pyplot(plt.gcf())
except FileNotFoundError:
    st.error("🚨 Error: `FashionDataset.csv` not found. Please make sure the file is in the same directory as `app.py`.")
    st.stop()
