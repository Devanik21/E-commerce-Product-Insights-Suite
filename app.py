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
import google.generativeai as genai
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="🛒 E-commerce Insights Suite", layout="wide")
st.title("🛒 E-commerce Product Insights Suite")
st.write("Exploring insights from the preloaded Fashion Dataset. Analyze segments, churn, sales trends, and more!")

# Load preloaded data
# Ensure 'FashionDataset.csv' is in the same directory as app.py or provide a full path.
try:
    df = pd.read_csv("FashionDataset.csv", index_col=0) # Use index_col=0 for the unnamed index
    st.subheader("🧾 Data Preview (FashionDataset.csv)")
    st.write(df.head())

    # Convert date columns
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            st.warning(f"Could not convert column '{col}' to datetime: {e}")

    # --- Sidebar for API Key and AI Model Info ---
    st.sidebar.subheader("✨ AI Configuration")
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", help="Get your API key from Google AI Studio.")
    st.sidebar.caption("Using AI Model: Gemini 1.5 Flash (via `gemini-1.5-flash-latest`)")
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Analysis Modules")

    # --- Main content with Tabs ---
    tab1, tab2 = st.tabs(["📊 Traditional Analysis", "🤖 AI Powered Insights"])

    with tab1:
        st.header("Traditional Analysis Modules")
        st.write("Select analysis options from the sidebar to view results here. Note: These modules expect specific column names (e.g., CustomerID, OrderDate, TotalAmount) which may not be present in the current FashionDataset.")

        # RFM Segmentation
        if st.sidebar.checkbox("🧩 RFM Segmentation", key="rfm_cb"):
            st.subheader("🧩 RFM Segmentation")
            if 'CustomerID' in df.columns and 'OrderDate' in df.columns and 'TotalAmount' in df.columns and not df['OrderDate'].isnull().all():
                snapshot_date = df['OrderDate'].max() + pd.Timedelta(days=1)
                rfm = df.groupby('CustomerID').agg({
                    'OrderDate': lambda x: (snapshot_date - x.max()).days,
                    'CustomerID': 'count',
                    'TotalAmount': 'sum'
                })
                rfm.columns = ['Recency', 'Frequency', 'Monetary']
                st.write(rfm.head())

                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(rfm)
                kmeans = KMeans(n_clusters=4, random_state=1, n_init='auto')
                rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
                st.write(rfm.groupby('Cluster').mean())
                fig, ax = plt.subplots()
                sns.pairplot(rfm.reset_index(), hue='Cluster', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("RFM analysis requires 'CustomerID', 'OrderDate', and 'TotalAmount' columns with valid data.")

        # Customer Churn Detection
        if st.sidebar.checkbox("📉 Customer Churn Detection", key="churn_cb"):
            st.subheader("📉 Customer Churn Detection")
            if 'CustomerID' in df.columns and 'OrderDate' in df.columns and not df['OrderDate'].isnull().all():
                recent_orders = df.groupby('CustomerID')['OrderDate'].max()
                churn_threshold = st.slider("Days since last order to be considered churned", 30, 180, 90)
                churned = recent_orders < (df['OrderDate'].max() - pd.Timedelta(days=churn_threshold))
                churn_summary = churned.value_counts().rename(index={True: 'Churned', False: 'Active'})
                st.bar_chart(churn_summary)
            else:
                st.warning("Churn detection requires 'CustomerID' and 'OrderDate' columns with valid data.")

        # Sales Forecasting
        if st.sidebar.checkbox("📈 Sales Forecasting", key="forecast_cb"):
            st.subheader("📈 Sales Forecasting (Simple Trend)")
            if 'OrderDate' in df.columns and 'TotalAmount' in df.columns and not df['OrderDate'].isnull().all():
                df_sorted = df.dropna(subset=['OrderDate', 'TotalAmount']).sort_values('OrderDate')
                if not df_sorted.empty:
                    df_grouped = df_sorted.groupby('OrderDate')['TotalAmount'].sum().reset_index()
                    df_grouped['Days'] = (df_grouped['OrderDate'] - df_grouped['OrderDate'].min()).dt.days
                    model = LinearRegression()
                    model.fit(df_grouped[['Days']], df_grouped['TotalAmount'])
                    future_days = np.arange(df_grouped['Days'].max() + 1, df_grouped['Days'].max() + 31).reshape(-1, 1)
                    future_sales = model.predict(future_days)
                    future_dates = [df_grouped['OrderDate'].max() + pd.Timedelta(days=i) for i in range(1, 31)]
                    forecast_df = pd.DataFrame({"OrderDate": future_dates, "ForecastedSales": future_sales})
                    
                    # Prepare data for st.line_chart
                    plot_df = pd.concat([
                        df_grouped[['OrderDate', 'TotalAmount']].rename(columns={"TotalAmount": "ActualSales"}).set_index('OrderDate'),
                        forecast_df.rename(columns={"ForecastedSales": "ForecastedSales"}).set_index('OrderDate')
                    ])
                    st.line_chart(plot_df)
                else:
                    st.warning("Not enough valid data for sales forecasting after filtering.")
            else:
                st.warning("Sales forecasting requires 'OrderDate' and 'TotalAmount' columns with valid data.")

        # Return Analysis
        if st.sidebar.checkbox("↩️ Return Analysis", key="return_cb"):
            st.subheader("↩️ Return Rate by Product")
            if 'ProductID' in df.columns and 'IsReturned' in df.columns:
                return_rate = df.groupby('ProductID')['IsReturned'].mean()
                st.bar_chart(return_rate.sort_values(ascending=False).head(10))
            else:
                st.warning("Return analysis requires 'ProductID' and 'IsReturned' columns.")

        # A/B Test Visual
        if st.sidebar.checkbox("🧪 A/B Test Summary", key="ab_cb"):
            st.subheader("🧪 A/B Test Summary for Pricing")
            if 'Group' in df.columns and 'Conversion' in df.columns:
                ab_summary = df.groupby('Group')['Conversion'].agg(['count', 'mean'])
                st.write(ab_summary)
                fig, ax = plt.subplots()
                sns.barplot(data=df, x='Group', y='Conversion', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("A/B Test summary requires 'Group' and 'Conversion' columns.")

    with tab2:
        st.header("🤖 AI Powered Insights")
        st.write("Use Gemini to generate content and analyze your fashion data.")

        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to use AI features.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')

                st.subheader("🛍️ Enhanced Product Description Generator")
                if not df.empty:
                    product_list = [f"{row['BrandName']} - {row['Deatils'][:50]}..." for index, row in df.iterrows()]
                    selected_product_display = st.selectbox("Select a product:", product_list, key="product_select")
                    
                    if selected_product_display:
                        selected_idx = product_list.index(selected_product_display)
                        product = df.iloc[selected_idx]

                        if st.button("✨ Generate Enhanced Description", key="gen_desc_btn"):
                            with st.spinner("Generating description..."):
                                prompt = f"""You are a fashion copywriter. Given the following product details, write an engaging and concise product description (around 50-100 words) suitable for an e-commerce website.
Product Details:
Brand: {product['BrandName']}
Details: {product['Deatils']}
Category: {product['Category']}
Original Price (MRP): {product['MRP']}
Selling Price: {product['SellPrice']}
Discount: {product['Discount']}
Available Sizes: {product['Sizes']}

Generate an enhanced product description:"""
                                try:
                                    response = model.generate_content(prompt)
                                    st.markdown(response.text)
                                except Exception as e:
                                    st.error(f"Error generating description: {e}")
                else:
                    st.info("No product data available to generate descriptions.")

                st.markdown("---")
                st.subheader("💬 Chat with Your Data (FashionDataset.csv)")
                user_question = st.text_area("Ask a question about the Fashion Dataset:", height=100, key="ai_question")
                if st.button("💬 Get Answer from AI", key="get_answer_btn"):
                    if user_question:
                        with st.spinner("Thinking..."):
                            # Provide a summary of the DataFrame as context
                            data_summary = f"""The dataset has the following columns: {df.columns.tolist()}.
Here are the first 3 rows of the data:
{df.head(3).to_string()}

Based on this data, please answer the question: "{user_question}"
If the question cannot be answered from the provided column names and sample data, please state that or ask for more specific information.
"""
                            try:
                                response = model.generate_content(data_summary)
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Error getting answer: {e}")
                    else:
                        st.warning("Please enter a question.")
            except Exception as e:
                st.error(f"Failed to configure or use Gemini API: {e}")
                st.info("Please ensure your API key is correct and has the necessary permissions.")

except FileNotFoundError:
    st.error("🚨 Error: `FashionDataset.csv` not found. Please make sure the file is in the same directory as `app.py`.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("🚨 Error: `FashionDataset.csv` is empty. Please provide a valid CSV file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()
