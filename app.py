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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Imports for potential advanced analytics tools (some may be placeholders)
import scipy.stats as stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
# import networkx as nx # For graph analysis
# import plotly.express as px # For advanced interactive visualizations
from lifelines import KaplanMeierFitter
import pymc as pm
import arviz as az
# import shap # For model interpretability
# from gensim.models import LdaModel # For topic modeling
# # import tensorflow as tf # For deep learning (conceptual)
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="🛒 E-commerce Sales Insights Suite", layout="wide")
st.title("🛒 E-commerce Sales Insights Suite")
st.write("Exploring insights from the preloaded Amazon Sales Report. Analyze segments, sales trends, and more!")

# Load preloaded data
# Ensure 'Amazon_Sale_Report_Sampled.csv' is in the same directory as app.py or provide a full path.
DATASET_FILENAME = "Amazon_Sale_Report_Sampled.csv"
try:
    df = pd.read_csv(DATASET_FILENAME, index_col=0) # Use index_col=0 for the first column as index
    df.columns = df.columns.str.strip() # Strip whitespace from column names
    if 'Unnamed: 22' in df.columns: # Drop common extraneous column from this dataset
        df = df.drop(columns=['Unnamed: 22'])

    st.subheader(f"🧾 Data Preview ({DATASET_FILENAME})")
    st.write(df.head())

    # Convert date columns
    # Specifically handle 'Date' column from Amazon dataset, then others generically
    date_column_to_format = 'Date' 
    if date_column_to_format in df.columns:
        try:
            df[date_column_to_format] = pd.to_datetime(df[date_column_to_format], format='%m-%d-%y', errors='coerce')
        except Exception as e:
            st.warning(f"Could not convert column '{date_column_to_format}' with specific format %m-%d-%y: {e}. Trying generic conversion.")
            df[date_column_to_format] = pd.to_datetime(df[date_column_to_format], errors='coerce')

    other_date_cols = [col for col in df.columns if 'date' in col.lower() and col != date_column_to_format]
    for col in other_date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist() # Update global list

    # --- Sidebar for API Key and AI Model Info ---
    st.sidebar.subheader("✨ AI Configuration")
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", help="Get your API key from Google AI Studio.")
    st.sidebar.caption("Using AI Model: Gemini 2.0 Flash (via `gemini-2.0-flash`)")
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Analysis Modules")
    # Traditional analysis options are now directly within Tab 1


    # --- Helper functions for column selection (can be used across tabs) ---
    def get_numeric_columns(data_frame):
        return data_frame.select_dtypes(include=np.number).columns.tolist()

    def get_categorical_columns(data_frame, nunique_threshold=30): # nunique_threshold helps filter out high-cardinality 'object' columns
        return [col for col in data_frame.columns if data_frame[col].nunique() < nunique_threshold and (data_frame[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data_frame[col]) or data_frame[col].dtype == 'bool')]


    # --- Main content with Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Traditional Analysis",
        "🤖 AI Powered Insights",
        "🔬 Advanced Analytics Toolkit"
    ])
    with tab1:
        st.header("Traditional Analysis Modules")
        st.write(f"Expand an analysis module below, configure the required columns from your '{DATASET_FILENAME}' dataset, and then click 'Run Analysis'.")

        # RFM Segmentation
        with st.expander("🧩 RFM Segmentation", expanded=False):
            # st.subheader("🧩 RFM Segmentation") # Subheader can be optional if expander title is clear
            st.info("RFM typically requires a Customer ID. With the Amazon dataset, you might use 'Order ID' as a proxy for unique transactions, or another column if you have customer identifiers. 'Date' for order date, and 'Amount' for monetary value.")
            
            # Dynamically get available columns
            all_cols = df.columns.tolist()
            numeric_cols_rfm = get_numeric_columns(df)
            date_cols_rfm = date_cols # Use globally defined date_cols

            customer_id_col_rfm = st.selectbox("Select Customer/Entity ID column for RFM:", all_cols, index=all_cols.index('Order ID') if 'Order ID' in all_cols else 0, key="rfm_cust_id")
            order_date_col_rfm = st.selectbox("Select Order Date column for RFM:", date_cols_rfm if date_cols_rfm else all_cols, index=date_cols_rfm.index('Date') if 'Date' in date_cols_rfm else (all_cols.index('Date') if 'Date' in all_cols else 0), key="rfm_date")
            total_amount_col_rfm = st.selectbox("Select Total Amount column for RFM:", numeric_cols_rfm if numeric_cols_rfm else all_cols, index=numeric_cols_rfm.index('Amount') if 'Amount' in numeric_cols_rfm else (all_cols.index('Amount') if 'Amount' in all_cols else 0), key="rfm_amount")

            # Analysis runs automatically if expander is open and columns are valid
            if customer_id_col_rfm and order_date_col_rfm and total_amount_col_rfm and \
               customer_id_col_rfm in df.columns and order_date_col_rfm in df.columns and \
               total_amount_col_rfm in df.columns and not df[order_date_col_rfm].isnull().all():
                
                rfm_df_copy = df[[customer_id_col_rfm, order_date_col_rfm, total_amount_col_rfm]].copy()
                rfm_df_copy[order_date_col_rfm] = pd.to_datetime(rfm_df_copy[order_date_col_rfm], errors='coerce')
                rfm_df_copy = rfm_df_copy.dropna(subset=[order_date_col_rfm, total_amount_col_rfm])

                if rfm_df_copy.empty:
                    st.warning("Not enough valid data for RFM analysis after filtering. Check selected columns and their data types.")
                else:
                    snapshot_date = rfm_df_copy[order_date_col_rfm].max() + pd.Timedelta(days=1)
                    rfm_agg = rfm_df_copy.groupby(customer_id_col_rfm).agg({
                        order_date_col_rfm: lambda x: (snapshot_date - x.max()).days,
                        customer_id_col_rfm: 'count', # Frequency based on selected ID
                        total_amount_col_rfm: 'sum'
                    })
                    rfm_agg.columns = ['Recency', 'Frequency', 'Monetary']
                    st.write("RFM Aggregated Data (Head):")
                    st.write(rfm_agg.head())

                    if len(rfm_agg) > 1 : # Need at least 2 samples for scaling and clustering
                        scaler = StandardScaler()
                        rfm_scaled = scaler.fit_transform(rfm_agg)
                        
                        n_clusters_rfm = min(4, len(rfm_agg)) # Adjust n_clusters if less data
                        if n_clusters_rfm > 1:
                            kmeans = KMeans(n_clusters=n_clusters_rfm, random_state=1, n_init='auto' if pd.__version__ >= '1.3' else 10)
                            rfm_agg['Cluster'] = kmeans.fit_predict(rfm_scaled)
                            st.write("RFM Cluster Means:")
                            st.write(rfm_agg.groupby('Cluster').mean())
                            
                            if len(rfm_agg) >= n_clusters_rfm : # Ensure enough data for pairplot
                                try:
                                    pair_fig = sns.pairplot(rfm_agg.reset_index(), hue='Cluster', vars=['Recency', 'Frequency', 'Monetary'])
                                    st.pyplot(pair_fig)
                                except Exception as e:
                                    st.warning(f"Could not generate pairplot for RFM: {e}")
                            else:
                                st.info("Not enough distinct data points to generate a meaningful pairplot after clustering.")
                        else:
                            st.warning("Not enough clusters can be formed (less than 2). RFM clustering skipped.")
                    else:
                        st.warning("Not enough unique entities for RFM clustering after aggregation.")
            elif any(col is None for col in [customer_id_col_rfm, order_date_col_rfm, total_amount_col_rfm]):
                st.info("Please select all required columns for RFM Analysis.")
            else: # This case handles when columns are selected but might be invalid (e.g., not in df or all NaNs for date)
                st.warning("RFM analysis requires valid 'Customer/Entity ID', 'Order Date', and 'Total Amount' columns. Ensure 'Order Date' column is correctly formatted as datetime and has data.")

        # Customer Churn Detection
        with st.expander("📉 Customer Churn Detection", expanded=False):
            # st.subheader("📉 Customer Churn Detection")
            st.info("Churn detection typically requires a Customer ID. With the Amazon dataset, you might use 'Order ID' or another identifier if available. 'Date' would be the order date.")
            
            all_cols_churn = df.columns.tolist()
            date_cols_churn = date_cols

            customer_id_col_churn = st.selectbox("Select Customer/Entity ID column for Churn:", all_cols_churn, index=all_cols_churn.index('Order ID') if 'Order ID' in all_cols_churn else 0, key="churn_cust_id")
            order_date_col_churn = st.selectbox("Select Order Date column for Churn:", date_cols_churn if date_cols_churn else all_cols_churn, index=date_cols_churn.index('Date') if 'Date' in date_cols_churn else (all_cols_churn.index('Date') if 'Date' in all_cols_churn else 0), key="churn_date")

            # Analysis runs automatically if expander is open and columns are valid
            if customer_id_col_churn and order_date_col_churn and \
               customer_id_col_churn in df.columns and order_date_col_churn in df.columns and \
               not df[order_date_col_churn].isnull().all():
                
                churn_df_copy = df[[customer_id_col_churn, order_date_col_churn]].copy()
                churn_df_copy[order_date_col_churn] = pd.to_datetime(churn_df_copy[order_date_col_churn], errors='coerce')
                churn_df_copy = churn_df_copy.dropna(subset=[order_date_col_churn])

                if churn_df_copy.empty:
                    st.warning("Not enough valid data for Churn analysis after filtering.")
                else:
                    recent_orders = churn_df_copy.groupby(customer_id_col_churn)[order_date_col_churn].max()
                    churn_threshold = st.slider("Days since last order to be considered churned", 30, 365, 90, key="churn_slider") # Slider remains interactive
                    churned = recent_orders < (churn_df_copy[order_date_col_churn].max() - pd.Timedelta(days=churn_threshold))
                    churn_summary = churned.value_counts(dropna=False).rename(index={True: 'Churned', False: 'Active', pd.NA: 'Unknown'})
                    st.write("Churn Summary:")
                    st.bar_chart(churn_summary)
            elif any(col is None for col in [customer_id_col_churn, order_date_col_churn]):
                st.info("Please select all required columns for Churn Detection.")
            else:
                st.warning("Churn detection requires valid 'Customer/Entity ID' and 'Order Date' columns. Ensure 'Order Date' column is correctly formatted as datetime and has data.")

        # Sales Forecasting
        with st.expander("📈 Sales Forecasting (Simple Trend)", expanded=False):
            # st.subheader("📈 Sales Forecasting (Simple Trend)")
            st.info("This uses a simple linear regression for forecasting. Select 'Date' for order date and 'Amount' for sales value.")

            all_cols_fc = df.columns.tolist()
            numeric_cols_fc = get_numeric_columns(df)
            date_cols_fc = date_cols

            order_date_col_fc = st.selectbox("Select Order Date column for Forecasting:", date_cols_fc if date_cols_fc else all_cols_fc, index=date_cols_fc.index('Date') if 'Date' in date_cols_fc else (all_cols_fc.index('Date') if 'Date' in all_cols_fc else 0), key="fc_date")
            total_amount_col_fc = st.selectbox("Select Total Amount column for Forecasting:", numeric_cols_fc if numeric_cols_fc else all_cols_fc, index=numeric_cols_fc.index('Amount') if 'Amount' in numeric_cols_fc else (all_cols_fc.index('Amount') if 'Amount' in all_cols_fc else 0), key="fc_amount")

            # Analysis runs automatically if expander is open and columns are valid
            if order_date_col_fc and total_amount_col_fc and \
               order_date_col_fc in df.columns and total_amount_col_fc in df.columns and \
               not df[order_date_col_fc].isnull().all():
                
                fc_df_copy = df[[order_date_col_fc, total_amount_col_fc]].copy()
                fc_df_copy[order_date_col_fc] = pd.to_datetime(fc_df_copy[order_date_col_fc], errors='coerce')
                df_sorted = fc_df_copy.dropna(subset=[order_date_col_fc, total_amount_col_fc]).sort_values(order_date_col_fc)

                if not df_sorted.empty and len(df_sorted) > 1:
                    df_grouped = df_sorted.groupby(order_date_col_fc)[total_amount_col_fc].sum().reset_index()
                    if len(df_grouped) > 1: # Need at least 2 points for linear regression
                        df_grouped['Days'] = (df_grouped[order_date_col_fc] - df_grouped[order_date_col_fc].min()).dt.days
                        model = LinearRegression()
                        model.fit(df_grouped[['Days']], df_grouped[total_amount_col_fc])
                        future_days_count = st.slider("Number of days to forecast:", 7, 90, 30, key="fc_days_slider") # Slider remains interactive
                        future_days = np.arange(df_grouped['Days'].max() + 1, df_grouped['Days'].max() + future_days_count + 1).reshape(-1, 1)
                        future_sales = model.predict(future_days)
                        future_dates = [df_grouped[order_date_col_fc].max() + pd.Timedelta(days=i) for i in range(1, future_days_count + 1)]
                        forecast_df = pd.DataFrame({order_date_col_fc: future_dates, "ForecastedSales": future_sales})
                        
                        plot_df = pd.concat([
                            df_grouped[[order_date_col_fc, total_amount_col_fc]].rename(columns={total_amount_col_fc: "ActualSales"}).set_index(order_date_col_fc),
                            forecast_df.rename(columns={"ForecastedSales": "ForecastedSales"}).set_index(order_date_col_fc)
                        ])
                        st.line_chart(plot_df)
                    else:
                        st.warning("Not enough unique date points after grouping for sales forecasting.")
                else:
                    st.warning("Not enough valid data for sales forecasting after filtering. Need at least 2 data points.")
            elif any(col is None for col in [order_date_col_fc, total_amount_col_fc]):
                st.info("Please select all required columns for Sales Forecasting.")
            else:
                st.warning("Sales forecasting requires valid 'Order Date' and 'Total Amount' columns. Ensure 'Order Date' column is correctly formatted as datetime and has data.")

        # Return Analysis
        with st.expander("↩️ Return Analysis", expanded=False):
            # st.subheader("↩️ Return Rate by Product")
            st.info(f"Return analysis requires a 'Product ID' (e.g., 'SKU', 'ASIN') and a 'Return Indicator' column (binary 0/1 or True/False). The 'Return Indicator' column is not present by default in the '{DATASET_FILENAME}'. You might need to create or join this data.")
            
            all_cols_ret = df.columns.tolist()
            product_id_col_ret = st.selectbox("Select Product ID column for Returns:", all_cols_ret, index=all_cols_ret.index('SKU') if 'SKU' in all_cols_ret else (all_cols_ret.index('ASIN') if 'ASIN' in all_cols_ret else 0), key="ret_pid")
            is_returned_col = st.selectbox("Select Return Indicator column (binary 0/1 or True/False):", [None] + all_cols_ret, index=0, key="ret_isret")
            
            # Analysis runs automatically if expander is open and columns are valid
            if product_id_col_ret and is_returned_col and is_returned_col in df.columns:
                try:
                    df_copy_ret = df[[product_id_col_ret, is_returned_col]].copy().dropna()
                    
                    # Attempt to convert to boolean/int if not already
                    if df_copy_ret[is_returned_col].dtype == 'object':
                        true_vals = ['true', 'yes', '1', 'returned', 'shipped - returned to seller'] 
                        false_vals = ['false', 'no', '0', 'not returned', 'shipped']
                        df_copy_ret[is_returned_col] = df_copy_ret[is_returned_col].astype(str).str.lower().map(lambda x: 1 if x in true_vals else (0 if x in false_vals else pd.NA))
                    
                    df_copy_ret[is_returned_col] = pd.to_numeric(df_copy_ret[is_returned_col], errors='coerce')
                    df_copy_ret = df_copy_ret.dropna(subset=[is_returned_col])

                    if not df_copy_ret[is_returned_col].isin([0,1]).all():
                        st.warning(f"Column '{is_returned_col}' must be binary (0/1 or True/False) after processing for return analysis. Please check its values.")
                    elif df_copy_ret.empty:
                        st.warning("No valid data for return analysis after processing the selected columns.")
                    else:
                        return_rate = df_copy_ret.groupby(product_id_col_ret)[is_returned_col].mean()
                        st.write("Top 10 Products by Return Rate:")
                        st.bar_chart(return_rate.sort_values(ascending=False).head(10))
                except Exception as e:
                    st.error(f"Could not process return indicator column '{is_returned_col}': {e}")
            elif product_id_col_ret and not is_returned_col: # Product ID selected, but return indicator is not
                 st.info(f"Please select a 'Return Indicator' column. This column is not present by default in the '{DATASET_FILENAME}'.")
            elif not product_id_col_ret and is_returned_col: # Return indicator selected, but product ID is not
                st.info("Please select a 'Product ID' column for Return Analysis.")
            elif not product_id_col_ret and not is_returned_col:
                st.info("Please select 'Product ID' and 'Return Indicator' columns for Return Analysis.")

        # A/B Test Visual
        with st.expander("🧪 A/B Test Summary", expanded=False):
            # st.subheader("🧪 A/B Test Summary")
            st.info(f"A/B testing requires specific 'Group' and 'Conversion' (binary outcome) columns. The '{DATASET_FILENAME}' may not have these directly. You can use categorical columns like 'Sales Channel' or 'Fulfilment' as groups, and 'B2B' status or a binarized numeric column as conversion.")
            
            all_cols_ab = df.columns.tolist()
            categorical_cols_ab = get_categorical_columns(df) 

            group_col_ab = st.selectbox("Select Group column for A/B Test (Categorical):", [None] + categorical_cols_ab, index=0, key="ab_group")
            conversion_col_ab_source = st.selectbox("Select column for Conversion/Outcome:", [None] + all_cols_ab, index=0, key="ab_conv_src")

            # Analysis runs automatically if expander is open and columns are valid
            if group_col_ab and conversion_col_ab_source and group_col_ab in df.columns and conversion_col_ab_source in df.columns:
                try:
                    df_copy_ab = df[[group_col_ab, conversion_col_ab_source]].copy().dropna()
                    conversion_col_final_name = "conversion_metric"
                    valid_conversion_for_ab = False

                    # Process conversion column to be binary
                    if df_copy_ab[conversion_col_ab_source].dtype == 'bool':
                        df_copy_ab[conversion_col_final_name] = df_copy_ab[conversion_col_ab_source].astype(int)
                        valid_conversion_for_ab = True
                    elif pd.api.types.is_numeric_dtype(df_copy_ab[conversion_col_ab_source]) and df_copy_ab[conversion_col_ab_source].nunique() == 2 and df_copy_ab[conversion_col_ab_source].isin([0,1]).all():
                        df_copy_ab[conversion_col_final_name] = df_copy_ab[conversion_col_ab_source]
                        valid_conversion_for_ab = True
                    elif pd.api.types.is_numeric_dtype(df_copy_ab[conversion_col_ab_source]):
                        st.write(f"Numeric column '{conversion_col_ab_source}' selected for conversion. Define a threshold to binarize it (values > threshold = 1, else 0).")
                        num_threshold_ab = st.number_input("Enter threshold:", value=df_copy_ab[conversion_col_ab_source].median(), key="ab_thresh")
                        df_copy_ab[conversion_col_final_name] = (df_copy_ab[conversion_col_ab_source] > num_threshold_ab).astype(int)
                        valid_conversion_for_ab = True
                    elif df_copy_ab[conversion_col_ab_source].dtype == 'object' or pd.api.types.is_categorical_dtype(df_copy_ab[conversion_col_ab_source]):
                        st.write(f"Categorical column '{conversion_col_ab_source}' selected for conversion. Select the 'positive' class to be treated as 1 (conversion).")
                        positive_class_ab = st.selectbox(f"Select positive class for '{conversion_col_ab_source}':", df_copy_ab[conversion_col_ab_source].unique(), key="ab_pos_class")
                        if positive_class_ab is not None:
                            df_copy_ab[conversion_col_final_name] = (df_copy_ab[conversion_col_ab_source] == positive_class_ab).astype(int)
                            valid_conversion_for_ab = True
                        else:
                            st.warning("Please select a positive class for the A/B test conversion for the analysis to proceed.")
                    else:
                        st.warning(f"Column '{conversion_col_ab_source}' is not easily convertible to a binary (0/1) conversion metric. Please choose a boolean, binary numeric, or categorical column with clear classes.")
                    
                    if valid_conversion_for_ab and conversion_col_final_name in df_copy_ab.columns:
                        if df_copy_ab[group_col_ab].nunique() < 2:
                            st.warning(f"The selected Group column '{group_col_ab}' has fewer than 2 unique groups. A/B testing requires at least two groups.")
                        else:
                            st.write(f"A/B Test Summary (Group: {group_col_ab}, Conversion: {conversion_col_final_name} from {conversion_col_ab_source})")
                            ab_summary = df_copy_ab.groupby(group_col_ab)[conversion_col_final_name].agg(['count', 'mean'])
                            ab_summary.columns = ['Total Count', 'Conversion Rate']
                            st.write(ab_summary)
                            
                            fig, ax = plt.subplots()
                            sns.barplot(data=df_copy_ab, x=group_col_ab, y=conversion_col_final_name, ax=ax, errorbar=None) 
                            ax.set_ylabel(f"Mean Conversion ({conversion_col_final_name})")
                            ax.set_title(f"Conversion Rate by {group_col_ab}")
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error during A/B Test summary: {e}")
            elif not group_col_ab or not conversion_col_ab_source:
                st.info("Please select both a 'Group' column and a 'Conversion/Outcome' column for A/B Test summary.")

        # Tool 1: Detailed Product Performance Analyzer
        with st.expander("📊 Detailed Product Performance Analyzer", expanded=False):
            st.info("Analyze product performance by revenue, quantity sold, and trends. Select relevant columns from your dataset.")

            all_cols_ppa = df.columns.tolist()
            numeric_cols_ppa = get_numeric_columns(df)
            date_cols_ppa = date_cols # Global date_cols

            st.markdown("#### Column Selection")
            col1_ppa, col2_ppa = st.columns(2)
            with col1_ppa:
                product_id_col_ppa = st.selectbox("Select Product ID column (e.g., SKU, ASIN):", all_cols_ppa, index=all_cols_ppa.index('SKU') if 'SKU' in all_cols_ppa else 0, key="ppa_pid")
                category_col_ppa = st.selectbox("Select Product Category column:", [None] + all_cols_ppa, index=([None] + all_cols_ppa).index('Category') if 'Category' in all_cols_ppa else 0, key="ppa_cat")
                date_col_ppa = st.selectbox("Select Date column:", date_cols_ppa if date_cols_ppa else all_cols_ppa, index=date_cols_ppa.index('Date') if 'Date' in date_cols_ppa else (all_cols_ppa.index('Date') if 'Date' in all_cols_ppa else 0), key="ppa_date")
            with col2_ppa:
                amount_col_ppa = st.selectbox("Select Sales Amount column:", numeric_cols_ppa if numeric_cols_ppa else all_cols_ppa, index=numeric_cols_ppa.index('Amount') if 'Amount' in numeric_cols_ppa else (all_cols_ppa.index('Amount') if 'Amount' in all_cols_ppa else 0), key="ppa_amount")
                qty_col_ppa = st.selectbox("Select Quantity Sold column:", numeric_cols_ppa if numeric_cols_ppa else all_cols_ppa, index=numeric_cols_ppa.index('Qty') if 'Qty' in numeric_cols_ppa else 0, key="ppa_qty")

            st.markdown("#### Filters & Options")
            top_n_ppa = st.slider("Select Top N products to display:", 5, 20, 10, key="ppa_top_n")
            
            # Date Range Filter
            min_date_ppa = df[date_col_ppa].min() if date_col_ppa and date_col_ppa in df.columns and not df[date_col_ppa].isnull().all() else datetime.now().date() - pd.Timedelta(days=365)
            max_date_ppa = df[date_col_ppa].max() if date_col_ppa and date_col_ppa in df.columns and not df[date_col_ppa].isnull().all() else datetime.now().date()
            
            # Ensure min_date_ppa and max_date_ppa are valid datetime objects for date_input
            if isinstance(min_date_ppa, pd.Timestamp): min_date_ppa = min_date_ppa.date()
            if isinstance(max_date_ppa, pd.Timestamp): max_date_ppa = max_date_ppa.date()

            start_date_ppa = st.date_input("Start date for PPA:", min_date_ppa, min_value=min_date_ppa, max_value=max_date_ppa, key="ppa_start_date")
            end_date_ppa = st.date_input("End date for PPA:", max_date_ppa, min_value=min_date_ppa, max_value=max_date_ppa, key="ppa_end_date")

            if start_date_ppa > end_date_ppa:
                st.warning("Start date cannot be after end date for Product Performance Analysis.")

            if st.button("🚀 Run Product Performance Analysis", key="ppa_run"):
                if not all([product_id_col_ppa, date_col_ppa, amount_col_ppa, qty_col_ppa]):
                    st.warning("Please select all required columns (Product ID, Date, Amount, Quantity).")
                elif start_date_ppa > end_date_ppa:
                    st.warning("Correct the date range before running analysis.")
                else:
                    try:
                        ppa_df = df.copy()
                        # Ensure date column is datetime
                        ppa_df[date_col_ppa] = pd.to_datetime(ppa_df[date_col_ppa], errors='coerce')
                        ppa_df = ppa_df.dropna(subset=[date_col_ppa, amount_col_ppa, qty_col_ppa, product_id_col_ppa])

                        # Apply date filter
                        ppa_df = ppa_df[(ppa_df[date_col_ppa] >= pd.to_datetime(start_date_ppa)) & (ppa_df[date_col_ppa] <= pd.to_datetime(end_date_ppa))]

                        if ppa_df.empty:
                            st.warning("No data available for the selected criteria in Product Performance Analysis.")
                        else:
                            st.subheader("Product Performance Results")

                            # 1. Top N Products by Revenue
                            top_products_revenue = ppa_df.groupby(product_id_col_ppa)[amount_col_ppa].sum().nlargest(top_n_ppa)
                            st.markdown(f"#### Top {top_n_ppa} Products by Revenue")
                            if not top_products_revenue.empty:
                                st.bar_chart(top_products_revenue)
                                st.dataframe(top_products_revenue.reset_index())
                            else:
                                st.info("No revenue data for top products.")

                            # 2. Top N Products by Quantity Sold
                            top_products_qty = ppa_df.groupby(product_id_col_ppa)[qty_col_ppa].sum().nlargest(top_n_ppa)
                            st.markdown(f"#### Top {top_n_ppa} Products by Quantity Sold")
                            if not top_products_qty.empty:
                                st.bar_chart(top_products_qty)
                                st.dataframe(top_products_qty.reset_index())
                            else:
                                st.info("No quantity data for top products.")

                            # 3. Sales Trend for a Selected Product
                            st.markdown("#### Sales Trend for a Specific Product")
                            product_options_ppa = ppa_df[product_id_col_ppa].unique().tolist()
                            if product_options_ppa:
                                selected_product_trend_ppa = st.selectbox("Select a product to see its sales trend:", product_options_ppa, index=0, key="ppa_trend_product")
                                if selected_product_trend_ppa:
                                    product_trend_df = ppa_df[ppa_df[product_id_col_ppa] == selected_product_trend_ppa].groupby(pd.Grouper(key=date_col_ppa, freq='M')).agg(
                                        TotalRevenue=(amount_col_ppa, 'sum'),
                                        TotalQuantity=(qty_col_ppa, 'sum')
                                    ).reset_index()
                                    if not product_trend_df.empty:
                                        st.line_chart(product_trend_df.set_index(date_col_ppa)[['TotalRevenue', 'TotalQuantity']])
                                    else:
                                        st.info(f"No sales trend data for product '{selected_product_trend_ppa}'.")
                            else:
                                st.info("No products available to select for trend analysis.")

                            # 4. Revenue Distribution by Category (if category column is selected)
                            if category_col_ppa and category_col_ppa in ppa_df.columns:
                                st.markdown(f"#### Revenue Distribution by {category_col_ppa}")
                                category_revenue = ppa_df.groupby(category_col_ppa)[amount_col_ppa].sum().sort_values(ascending=False)
                                if not category_revenue.empty:
                                    fig_cat_rev, ax_cat_rev = plt.subplots()
                                    category_revenue.head(10).plot(kind='bar', ax=ax_cat_rev) # Top 10 categories
                                    ax_cat_rev.set_ylabel("Total Revenue")
                                    ax_cat_rev.set_title(f"Top 10 {category_col_ppa} by Revenue")
                                    plt.xticks(rotation=45, ha="right")
                                    plt.tight_layout()
                                    st.pyplot(fig_cat_rev)
                                    st.dataframe(category_revenue.reset_index().head(20)) # Show more in table
                                else:
                                    st.info(f"No revenue data for category '{category_col_ppa}'.")

                            # 5. Average Selling Price (ASP) Analysis
                            st.markdown("#### Average Selling Price (ASP) per Product")
                            ppa_df['ASP'] = ppa_df[amount_col_ppa] / ppa_df[qty_col_ppa].replace(0, np.nan) # Avoid division by zero
                            asp_analysis = ppa_df.groupby(product_id_col_ppa)['ASP'].mean().nlargest(top_n_ppa)
                            if not asp_analysis.empty:
                                st.write(f"Top {top_n_ppa} products by Average Selling Price (mean over transactions):")
                                st.dataframe(asp_analysis.reset_index())
                            else:
                                st.info("Could not calculate ASP or no data available.")

                    except Exception as e:
                        st.error(f"An error occurred during Product Performance Analysis: {e}")

        # Tool 2: Geographic Sales Insights
        with st.expander("🌍 Geographic Sales Insights", expanded=False):
            st.info("Analyze sales performance across different geographic regions. Select relevant location and sales columns.")

            all_cols_geo = df.columns.tolist()
            numeric_cols_geo = get_numeric_columns(df)
            date_cols_geo = date_cols

            st.markdown("#### Column Selection")
            col1_geo, col2_geo = st.columns(2)
            with col1_geo:
                date_col_geo = st.selectbox("Select Date column:", date_cols_geo if date_cols_geo else all_cols_geo, index=date_cols_geo.index('Date') if 'Date' in date_cols_geo else (all_cols_geo.index('Date') if 'Date' in all_cols_geo else 0), key="geo_date")
                amount_col_geo = st.selectbox("Select Sales Amount column:", numeric_cols_geo if numeric_cols_geo else all_cols_geo, index=numeric_cols_geo.index('Amount') if 'Amount' in numeric_cols_geo else (all_cols_geo.index('Amount') if 'Amount' in all_cols_geo else 0), key="geo_amount")
                order_id_col_geo = st.selectbox("Select Order ID column (for order counts):", all_cols_geo, index=all_cols_geo.index('Order ID') if 'Order ID' in all_cols_geo else 0, key="geo_order_id")
            with col2_geo:
                state_col_geo = st.selectbox("Select State column:", [None] + all_cols_geo, index=([None] + all_cols_geo).index('ship-state') if 'ship-state' in all_cols_geo else 0, key="geo_state")
                city_col_geo = st.selectbox("Select City column:", [None] + all_cols_geo, index=([None] + all_cols_geo).index('ship-city') if 'ship-city' in all_cols_geo else 0, key="geo_city")
                country_col_geo = st.selectbox("Select Country column:", [None] + all_cols_geo, index=([None] + all_cols_geo).index('ship-country') if 'ship-country' in all_cols_geo else 0, key="geo_country")

            st.markdown("#### Filters & Options")
            top_n_geo = st.slider("Select Top N locations to display:", 5, 20, 10, key="geo_top_n")
            
            min_date_geo = df[date_col_geo].min() if date_col_geo and date_col_geo in df.columns and not df[date_col_geo].isnull().all() else datetime.now().date() - pd.Timedelta(days=365)
            max_date_geo = df[date_col_geo].max() if date_col_geo and date_col_geo in df.columns and not df[date_col_geo].isnull().all() else datetime.now().date()
            if isinstance(min_date_geo, pd.Timestamp): min_date_geo = min_date_geo.date()
            if isinstance(max_date_geo, pd.Timestamp): max_date_geo = max_date_geo.date()

            start_date_geo = st.date_input("Start date for Geo Analysis:", min_date_geo, min_value=min_date_geo, max_value=max_date_geo, key="geo_start_date")
            end_date_geo = st.date_input("End date for Geo Analysis:", max_date_geo, min_value=min_date_geo, max_value=max_date_geo, key="geo_end_date")

            if start_date_geo > end_date_geo:
                st.warning("Start date cannot be after end date for Geographic Sales Analysis.")

            if st.button("🗺️ Run Geographic Sales Analysis", key="geo_run"):
                if not all([date_col_geo, amount_col_geo, order_id_col_geo]):
                    st.warning("Please select Date, Amount, and Order ID columns.")
                elif start_date_geo > end_date_geo:
                    st.warning("Correct the date range before running analysis.")
                elif not any([state_col_geo, city_col_geo, country_col_geo]):
                    st.warning("Please select at least one geographic column (State, City, or Country).")
                else:
                    try:
                        geo_df = df.copy()
                        geo_df[date_col_geo] = pd.to_datetime(geo_df[date_col_geo], errors='coerce')
                        geo_df = geo_df.dropna(subset=[date_col_geo, amount_col_geo, order_id_col_geo])
                        geo_df = geo_df[(geo_df[date_col_geo] >= pd.to_datetime(start_date_geo)) & (geo_df[date_col_geo] <= pd.to_datetime(end_date_geo))]

                        if geo_df.empty:
                            st.warning("No data available for the selected criteria in Geographic Sales Analysis.")
                        else:
                            st.subheader("Geographic Sales Results")

                            if country_col_geo and country_col_geo in geo_df.columns:
                                st.markdown(f"#### Sales by {country_col_geo}")
                                country_sales = geo_df.groupby(country_col_geo).agg(
                                    TotalRevenue=(amount_col_geo, 'sum'),
                                    TotalOrders=(order_id_col_geo, 'nunique')
                                ).nlargest(top_n_geo, 'TotalRevenue')
                                if not country_sales.empty:
                                    st.bar_chart(country_sales['TotalRevenue'])
                                    st.dataframe(country_sales)
                                else:
                                    st.info(f"No sales data by {country_col_geo}.")

                            if state_col_geo and state_col_geo in geo_df.columns:
                                st.markdown(f"#### Top {top_n_geo} {state_col_geo} by Revenue & Orders")
                                state_sales = geo_df.groupby(state_col_geo).agg(
                                    TotalRevenue=(amount_col_geo, 'sum'),
                                    TotalOrders=(order_id_col_geo, 'nunique')
                                ).nlargest(top_n_geo, 'TotalRevenue')
                                if not state_sales.empty:
                                    st.bar_chart(state_sales['TotalRevenue'])
                                    st.dataframe(state_sales)
                                else:
                                    st.info(f"No sales data by {state_col_geo}.")

                            if city_col_geo and city_col_geo in geo_df.columns:
                                st.markdown(f"#### Top {top_n_geo} {city_col_geo} by Revenue & Orders")
                                # Optional: Filter cities by a selected state
                                if state_col_geo and state_col_geo in geo_df.columns:
                                    available_states_geo = geo_df[state_col_geo].dropna().unique().tolist()
                                    selected_state_filter_geo = st.selectbox(f"Filter cities by {state_col_geo} (optional):", [None] + available_states_geo, key="geo_city_state_filter")
                                    if selected_state_filter_geo:
                                        city_df_filtered = geo_df[geo_df[state_col_geo] == selected_state_filter_geo]
                                    else:
                                        city_df_filtered = geo_df
                                else:
                                    city_df_filtered = geo_df
                                
                                city_sales = city_df_filtered.groupby(city_col_geo).agg(
                                    TotalRevenue=(amount_col_geo, 'sum'),
                                    TotalOrders=(order_id_col_geo, 'nunique')
                                ).nlargest(top_n_geo, 'TotalRevenue')
                                if not city_sales.empty:
                                    st.bar_chart(city_sales['TotalRevenue'])
                                    st.dataframe(city_sales)
                                else:
                                    st.info(f"No sales data by {city_col_geo} for the current selection.")
                    except Exception as e:
                        st.error(f"An error occurred during Geographic Sales Analysis: {e}")

        # Tool 3: Sales Channel & Fulfilment Analysis
        with st.expander("🚚 Sales Channel & Fulfilment Analysis", expanded=False):
            st.info("Compare performance across different sales channels and fulfilment methods.")

            all_cols_scf = df.columns.tolist()
            numeric_cols_scf = get_numeric_columns(df)
            date_cols_scf = date_cols

            st.markdown("#### Column Selection")
            col1_scf, col2_scf = st.columns(2)
            with col1_scf:
                date_col_scf = st.selectbox("Select Date column:", date_cols_scf if date_cols_scf else all_cols_scf, index=date_cols_scf.index('Date') if 'Date' in date_cols_scf else (all_cols_scf.index('Date') if 'Date' in all_cols_scf else 0), key="scf_date")
                amount_col_scf = st.selectbox("Select Sales Amount column:", numeric_cols_scf if numeric_cols_scf else all_cols_scf, index=numeric_cols_scf.index('Amount') if 'Amount' in numeric_cols_scf else (all_cols_scf.index('Amount') if 'Amount' in all_cols_scf else 0), key="scf_amount")
            with col2_scf:
                order_id_col_scf = st.selectbox("Select Order ID column:", all_cols_scf, index=all_cols_scf.index('Order ID') if 'Order ID' in all_cols_scf else 0, key="scf_order_id")
                sales_channel_col_scf = st.selectbox("Select Sales Channel column:", [None] + all_cols_scf, index=([None] + all_cols_scf).index('Sales Channel') if 'Sales Channel' in all_cols_scf else 0, key="scf_sales_channel")
                fulfilment_col_scf = st.selectbox("Select Fulfilment column:", [None] + all_cols_scf, index=([None] + all_cols_scf).index('Fulfilment') if 'Fulfilment' in all_cols_scf else 0, key="scf_fulfilment")

            st.markdown("#### Filters")
            min_date_scf = df[date_col_scf].min() if date_col_scf and date_col_scf in df.columns and not df[date_col_scf].isnull().all() else datetime.now().date() - pd.Timedelta(days=365)
            max_date_scf = df[date_col_scf].max() if date_col_scf and date_col_scf in df.columns and not df[date_col_scf].isnull().all() else datetime.now().date()
            if isinstance(min_date_scf, pd.Timestamp): min_date_scf = min_date_scf.date()
            if isinstance(max_date_scf, pd.Timestamp): max_date_scf = max_date_scf.date()

            start_date_scf = st.date_input("Start date for SCF Analysis:", min_date_scf, min_value=min_date_scf, max_value=max_date_scf, key="scf_start_date")
            end_date_scf = st.date_input("End date for SCF Analysis:", max_date_scf, min_value=min_date_scf, max_value=max_date_scf, key="scf_end_date")

            if start_date_scf > end_date_scf:
                st.warning("Start date cannot be after end date for Sales Channel & Fulfilment Analysis.")

            if st.button("📊 Run Sales Channel & Fulfilment Analysis", key="scf_run"):
                if not all([date_col_scf, amount_col_scf, order_id_col_scf]):
                    st.warning("Please select Date, Amount, and Order ID columns.")
                elif start_date_scf > end_date_scf:
                    st.warning("Correct the date range before running analysis.")
                elif not (sales_channel_col_scf or fulfilment_col_scf):
                    st.warning("Please select at least a Sales Channel or Fulfilment column.")
                else:
                    try:
                        scf_df = df.copy()
                        scf_df[date_col_scf] = pd.to_datetime(scf_df[date_col_scf], errors='coerce')
                        scf_df = scf_df.dropna(subset=[date_col_scf, amount_col_scf, order_id_col_scf])
                        scf_df = scf_df[(scf_df[date_col_scf] >= pd.to_datetime(start_date_scf)) & (scf_df[date_col_scf] <= pd.to_datetime(end_date_scf))]

                        if scf_df.empty:
                            st.warning("No data available for the selected criteria in Sales Channel & Fulfilment Analysis.")
                        else:
                            st.subheader("Sales Channel & Fulfilment Results")

                            if sales_channel_col_scf and sales_channel_col_scf in scf_df.columns:
                                st.markdown(f"#### Performance by {sales_channel_col_scf}")
                                sc_perf = scf_df.groupby(sales_channel_col_scf).agg(
                                    TotalRevenue=(amount_col_scf, 'sum'),
                                    TotalOrders=(order_id_col_scf, 'nunique'),
                                    AverageOrderValue=(amount_col_scf, lambda x: x.sum() / scf_df.loc[x.index, order_id_col_scf].nunique() if scf_df.loc[x.index, order_id_col_scf].nunique() > 0 else 0)
                                ).sort_values(by="TotalRevenue", ascending=False)
                                if not sc_perf.empty:
                                    st.dataframe(sc_perf)
                                    st.bar_chart(sc_perf[['TotalRevenue', 'TotalOrders']])
                                else:
                                    st.info(f"No data for {sales_channel_col_scf}.")

                            if fulfilment_col_scf and fulfilment_col_scf in scf_df.columns:
                                st.markdown(f"#### Performance by {fulfilment_col_scf}")
                                ff_perf = scf_df.groupby(fulfilment_col_scf).agg(
                                    TotalRevenue=(amount_col_scf, 'sum'),
                                    TotalOrders=(order_id_col_scf, 'nunique'),
                                    AverageOrderValue=(amount_col_scf, lambda x: x.sum() / scf_df.loc[x.index, order_id_col_scf].nunique() if scf_df.loc[x.index, order_id_col_scf].nunique() > 0 else 0)
                                ).sort_values(by="TotalRevenue", ascending=False)
                                if not ff_perf.empty:
                                    st.dataframe(ff_perf)
                                    st.bar_chart(ff_perf[['TotalRevenue', 'TotalOrders']])
                                else:
                                    st.info(f"No data for {fulfilment_col_scf}.")

                            if sales_channel_col_scf and sales_channel_col_scf in scf_df.columns and fulfilment_col_scf and fulfilment_col_scf in scf_df.columns:
                                st.markdown(f"#### Revenue: {sales_channel_col_scf} vs. {fulfilment_col_scf}")
                                cross_tab_rev = pd.crosstab(index=scf_df[sales_channel_col_scf], columns=scf_df[fulfilment_col_scf], values=scf_df[amount_col_scf], aggfunc='sum').fillna(0)
                                if not cross_tab_rev.empty:
                                    st.dataframe(cross_tab_rev)
                                    fig_ct_rev, ax_ct_rev = plt.subplots()
                                    sns.heatmap(cross_tab_rev, annot=True, fmt=".0f", cmap="viridis", ax=ax_ct_rev)
                                    ax_ct_rev.set_title(f"Revenue by {sales_channel_col_scf} and {fulfilment_col_scf}")
                                    st.pyplot(fig_ct_rev)
                                else:
                                    st.info("No data for cross-tabulation.")
                    except Exception as e:
                        st.error(f"An error occurred during Sales Channel & Fulfilment Analysis: {e}")

        # Tool 4: Order Characteristics Analysis
        with st.expander("📦 Order Characteristics Analysis", expanded=False):
            st.info("Analyze characteristics of orders, such as size, value, and B2B status.")

            all_cols_oca = df.columns.tolist()
            numeric_cols_oca = get_numeric_columns(df)
            date_cols_oca = date_cols

            st.markdown("#### Column Selection")
            col1_oca, col2_oca = st.columns(2)
            with col1_oca:
                date_col_oca = st.selectbox("Select Date column:", date_cols_oca if date_cols_oca else all_cols_oca, index=date_cols_oca.index('Date') if 'Date' in date_cols_oca else (all_cols_oca.index('Date') if 'Date' in all_cols_oca else 0), key="oca_date")
                amount_col_oca = st.selectbox("Select Order Amount column:", numeric_cols_oca if numeric_cols_oca else all_cols_oca, index=numeric_cols_oca.index('Amount') if 'Amount' in numeric_cols_oca else (all_cols_oca.index('Amount') if 'Amount' in all_cols_oca else 0), key="oca_amount")
                qty_col_oca = st.selectbox("Select Order Quantity column:", numeric_cols_oca if numeric_cols_oca else all_cols_oca, index=numeric_cols_oca.index('Qty') if 'Qty' in numeric_cols_oca else 0, key="oca_qty")
            with col2_oca:
                order_id_col_oca = st.selectbox("Select Order ID column:", all_cols_oca, index=all_cols_oca.index('Order ID') if 'Order ID' in all_cols_oca else 0, key="oca_order_id")
                b2b_col_oca = st.selectbox("Select B2B indicator column (boolean/binary):", [None] + all_cols_oca, index=([None] + all_cols_oca).index('B2B') if 'B2B' in all_cols_oca else 0, key="oca_b2b")

            st.markdown("#### Filters")
            min_date_oca = df[date_col_oca].min() if date_col_oca and date_col_oca in df.columns and not df[date_col_oca].isnull().all() else datetime.now().date() - pd.Timedelta(days=365)
            max_date_oca = df[date_col_oca].max() if date_col_oca and date_col_oca in df.columns and not df[date_col_oca].isnull().all() else datetime.now().date()
            if isinstance(min_date_oca, pd.Timestamp): min_date_oca = min_date_oca.date()
            if isinstance(max_date_oca, pd.Timestamp): max_date_oca = max_date_oca.date()

            start_date_oca = st.date_input("Start date for OCA:", min_date_oca, min_value=min_date_oca, max_value=max_date_oca, key="oca_start_date")
            end_date_oca = st.date_input("End date for OCA:", max_date_oca, min_value=min_date_oca, max_value=max_date_oca, key="oca_end_date")

            if start_date_oca > end_date_oca:
                st.warning("Start date cannot be after end date for Order Characteristics Analysis.")

            if st.button("🔍 Run Order Characteristics Analysis", key="oca_run"):
                if not all([date_col_oca, amount_col_oca, qty_col_oca, order_id_col_oca]):
                    st.warning("Please select Date, Amount, Quantity, and Order ID columns.")
                elif start_date_oca > end_date_oca:
                    st.warning("Correct the date range before running analysis.")
                else:
                    try:
                        oca_df = df.copy()
                        oca_df[date_col_oca] = pd.to_datetime(oca_df[date_col_oca], errors='coerce')
                        oca_df = oca_df.dropna(subset=[date_col_oca, amount_col_oca, qty_col_oca, order_id_col_oca])
                        oca_df = oca_df[(oca_df[date_col_oca] >= pd.to_datetime(start_date_oca)) & (oca_df[date_col_oca] <= pd.to_datetime(end_date_oca))]

                        if oca_df.empty:
                            st.warning("No data available for the selected criteria in Order Characteristics Analysis.")
                        else:
                            st.subheader("Order Characteristics Results")

                            # Order-level aggregation
                            order_summary_df = oca_df.groupby(order_id_col_oca).agg(
                                TotalOrderAmount=(amount_col_oca, 'sum'),
                                TotalOrderQuantity=(qty_col_oca, 'sum'),
                                IsB2B=(b2b_col_oca, 'first') if b2b_col_oca and b2b_col_oca in oca_df.columns else (b2b_col_oca, lambda x: None) # Handle if B2B not selected
                            ).reset_index()

                            st.markdown("#### Distribution of Order Values (Amount per Order)")
                            fig_ova, ax_ova = plt.subplots()
                            sns.histplot(order_summary_df['TotalOrderAmount'], kde=True, ax=ax_ova)
                            ax_ova.set_title("Distribution of Total Order Amount")
                            ax_ova.set_xlabel("Total Amount per Order")
                            st.pyplot(fig_ova)
                            st.write(order_summary_df['TotalOrderAmount'].describe())

                            st.markdown("#### Distribution of Order Sizes (Quantity per Order)")
                            fig_oqa, ax_oqa = plt.subplots()
                            sns.histplot(order_summary_df['TotalOrderQuantity'], kde=True, ax=ax_oqa, binwidth=max(1, int(order_summary_df['TotalOrderQuantity'].max()/20)))
                            ax_oqa.set_title("Distribution of Total Order Quantity")
                            ax_oqa.set_xlabel("Total Quantity per Order")
                            st.pyplot(fig_oqa)
                            st.write(order_summary_df['TotalOrderQuantity'].describe())

                            if b2b_col_oca and b2b_col_oca in oca_df.columns:
                                st.markdown(f"#### B2B vs. Non-B2B Analysis (based on '{b2b_col_oca}')")
                                # Ensure B2B column is boolean-like or can be interpreted
                                if order_summary_df['IsB2B'].isnull().all() or order_summary_df['IsB2B'].nunique() < 2 :
                                    st.info(f"The B2B column '{b2b_col_oca}' does not have enough distinct values (True/False or 1/0) for comparison or is mostly empty.")
                                else:
                                    # Attempt to convert to boolean if it's not already
                                    try:
                                        order_summary_df['IsB2B_bool'] = order_summary_df['IsB2B'].astype(bool)
                                    except: # Fallback for strings like 'Yes'/'No'
                                        true_vals = [True, 1, 'true', 'yes', 'y', 'b2b']
                                        order_summary_df['IsB2B_bool'] = order_summary_df['IsB2B'].astype(str).str.lower().isin(true_vals)

                                    b2b_analysis = order_summary_df.groupby('IsB2B_bool').agg(
                                        TotalRevenue=('TotalOrderAmount', 'sum'),
                                        NumberOfOrders=(order_id_col_oca, 'count'),
                                        AverageOrderValue=('TotalOrderAmount', 'mean'),
                                        AverageOrderQuantity=('TotalOrderQuantity', 'mean')
                                    )
                                    b2b_analysis.index.name = f'{b2b_col_oca} Status (True=B2B)'
                                    st.dataframe(b2b_analysis)

                                    fig_b2b_rev, ax_b2b_rev = plt.subplots()
                                    b2b_analysis['TotalRevenue'].plot(kind='pie', autopct='%1.1f%%', ax=ax_b2b_rev, title='Revenue by B2B Status')
                                    st.pyplot(fig_b2b_rev)
                            else:
                                st.info("B2B column not selected for B2B vs. Non-B2B analysis.")
                    except Exception as e:
                        st.error(f"An error occurred during Order Characteristics Analysis: {e}")

        # Tool 5: Promotion Insights (Conditional)
        with st.expander("🎉 Promotion Insights (Based on 'promotion-ids')", expanded=False):
            st.info("Analyze the impact of promotions. This tool relies on a 'promotion-ids' column. The effectiveness depends on how this column is populated in your data (e.g., non-empty for promoted items).")

            all_cols_promo = df.columns.tolist()
            numeric_cols_promo = get_numeric_columns(df)
            date_cols_promo = date_cols

            st.markdown("#### Column Selection")
            col1_promo, col2_promo = st.columns(2)
            with col1_promo:
                date_col_promo = st.selectbox("Select Date column:", date_cols_promo if date_cols_promo else all_cols_promo, index=date_cols_promo.index('Date') if 'Date' in date_cols_promo else (all_cols_promo.index('Date') if 'Date' in all_cols_promo else 0), key="promo_date")
                amount_col_promo = st.selectbox("Select Sales Amount column:", numeric_cols_promo if numeric_cols_promo else all_cols_promo, index=numeric_cols_promo.index('Amount') if 'Amount' in numeric_cols_promo else (all_cols_promo.index('Amount') if 'Amount' in all_cols_promo else 0), key="promo_amount")
            with col2_promo:
                qty_col_promo = st.selectbox("Select Quantity Sold column:", numeric_cols_promo if numeric_cols_promo else all_cols_promo, index=numeric_cols_promo.index('Qty') if 'Qty' in numeric_cols_promo else 0, key="promo_qty")
                promo_ids_col = st.selectbox("Select Promotion IDs column:", [None] + all_cols_promo, index=([None] + all_cols_promo).index('promotion-ids') if 'promotion-ids' in all_cols_promo else 0, key="promo_ids")

            st.markdown("#### Filters")
            min_date_promo = df[date_col_promo].min() if date_col_promo and date_col_promo in df.columns and not df[date_col_promo].isnull().all() else datetime.now().date() - pd.Timedelta(days=365)
            max_date_promo = df[date_col_promo].max() if date_col_promo and date_col_promo in df.columns and not df[date_col_promo].isnull().all() else datetime.now().date()
            if isinstance(min_date_promo, pd.Timestamp): min_date_promo = min_date_promo.date()
            if isinstance(max_date_promo, pd.Timestamp): max_date_promo = max_date_promo.date()

            start_date_promo = st.date_input("Start date for Promo Analysis:", min_date_promo, min_value=min_date_promo, max_value=max_date_promo, key="promo_start_date")
            end_date_promo = st.date_input("End date for Promo Analysis:", max_date_promo, min_value=min_date_promo, max_value=max_date_promo, key="promo_end_date")

            if start_date_promo > end_date_promo:
                st.warning("Start date cannot be after end date for Promotion Insights.")

            if st.button("💡 Run Promotion Insights Analysis", key="promo_run"):
                if not all([date_col_promo, amount_col_promo, qty_col_promo, promo_ids_col]):
                    st.warning("Please select Date, Amount, Quantity, and Promotion IDs columns.")
                elif start_date_promo > end_date_promo:
                    st.warning("Correct the date range before running analysis.")
                else:
                    try:
                        promo_df = df.copy()
                        promo_df[date_col_promo] = pd.to_datetime(promo_df[date_col_promo], errors='coerce')
                        promo_df = promo_df.dropna(subset=[date_col_promo, amount_col_promo, qty_col_promo]) # Promo ID can be NaN
                        promo_df = promo_df[(promo_df[date_col_promo] >= pd.to_datetime(start_date_promo)) & (promo_df[date_col_promo] <= pd.to_datetime(end_date_promo))]

                        if promo_df.empty:
                            st.warning("No data available for the selected criteria in Promotion Insights.")
                        else:
                            st.subheader("Promotion Insights Results")

                            # Define 'HasPromotion' based on whether promo_ids_col is NaN or empty
                            promo_df['HasPromotion'] = ~promo_df[promo_ids_col].isnull() & (promo_df[promo_ids_col].astype(str).str.strip() != '')

                            st.markdown("#### Orders With vs. Without Promotions")
                            promo_summary = promo_df['HasPromotion'].value_counts().rename(index={True: 'With Promotion', False: 'Without Promotion'})
                            st.bar_chart(promo_summary)
                            st.write(promo_summary)

                            if promo_df['HasPromotion'].any(): # If there are any promoted items
                                st.markdown("#### Performance Comparison: Promoted vs. Non-Promoted Items/Orders")
                                # Note: This is item-level. For order-level, would need to group by Order ID first.
                                # For simplicity, let's do item-level comparison here.
                                promo_comparison = promo_df.groupby('HasPromotion').agg(
                                    TotalRevenue=(amount_col_promo, 'sum'),
                                    TotalQuantity=(qty_col_promo, 'sum'),
                                    AverageItemPrice=(amount_col_promo, 'mean'), # Avg price of items in this group
                                    AverageItemQuantity=(qty_col_promo, 'mean'), # Avg qty of items in this group
                                    NumberOfTransactions=('Order ID', 'count') # Assuming 'Order ID' is present for transaction count
                                ).rename(index={True: 'With Promotion', False: 'Without Promotion'})
                                st.dataframe(promo_comparison)

                                fig_promo_rev, ax_promo_rev = plt.subplots()
                                promo_comparison['TotalRevenue'].plot(kind='bar', ax=ax_promo_rev, title='Total Revenue: Promoted vs. Non-Promoted')
                                st.pyplot(fig_promo_rev)

                                # Top Promotion IDs by Revenue (if IDs are meaningful and not too many)
                                if promo_df[promo_df['HasPromotion']][promo_ids_col].nunique() > 0 and promo_df[promo_df['HasPromotion']][promo_ids_col].nunique() < 100: # Avoid too many unique IDs
                                    st.markdown(f"#### Top Promotion IDs by Revenue (from '{promo_ids_col}')")
                                    top_promo_codes = promo_df[promo_df['HasPromotion']].groupby(promo_ids_col)[amount_col_promo].sum().nlargest(10)
                                    if not top_promo_codes.empty:
                                        st.bar_chart(top_promo_codes)
                                        st.dataframe(top_promo_codes.reset_index())
                                    else:
                                        st.info("No specific promotion IDs found or no revenue associated after filtering.")
                                else:
                                    st.info(f"Promotion ID column '{promo_ids_col}' has too many unique values or no promoted items to list top IDs effectively.")
                            else:
                                st.info("No items marked with promotions found in the selected data based on the Promotion IDs column.")
                    except Exception as e:
                        st.error(f"An error occurred during Promotion Insights Analysis: {e}")

    with tab2:
        st.header("🤖 AI Powered Insights")
        st.write(f"Use Gemini to generate content and analyze your '{DATASET_FILENAME}' data.")

        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to use AI features.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')

                st.subheader("🛍️ Enhanced Product Description Generator")
                if not df.empty:
                    # Create a list of products for selection, ensuring columns exist
                    product_list_data = []
                    # Use a limited number of rows for performance in selectbox if df is large
                    df_sample_desc = df.head(100) if len(df) > 100 else df 
                    for index, row in df_sample_desc.iterrows():
                        sku = row.get('SKU', 'N/A')
                        style = row.get('Style', 'N/A')
                        category = row.get('Category', 'N/A')
                        product_list_data.append(f"{sku} - {style} ({category})")
                    
                    if not product_list_data:
                        st.info("No products to display for description generation (check SKU, Style, Category columns).")
                    else:
                        selected_product_display = st.selectbox("Select a product (from first 100 rows):", product_list_data, key="product_select_ai")
                        
                        if selected_product_display:
                            # Find the original index in df_sample_desc, then use that to get from original df if needed,
                            # or just use the df_sample_desc row. For simplicity, using df_sample_desc.
                            selected_idx_desc = product_list_data.index(selected_product_display)
                            product = df_sample_desc.iloc[selected_idx_desc]

                            if st.button("✨ Generate Enhanced Description", key="gen_desc_btn"):
                                with st.spinner("Generating description..."):
                                    prompt = f"""You are a fashion and e-commerce copywriter. Given the following product details from an Amazon sales report, write an engaging and concise product description (around 50-100 words) suitable for an e-commerce website. Highlight key features.
    Product Details:
    SKU: {product.get('SKU', 'N/A')}
    Style: {product.get('Style', 'N/A')}
    Category: {product.get('Category', 'N/A')}
    Size: {product.get('Size', 'N/A')}
    Selling Price: {product.get('Amount', 'N/A')} {product.get('currency', '')}
    Sales Channel: {product.get('Sales Channel', 'N/A')}
    Fulfilment: {product.get('Fulfilment', 'N/A')}

    Generate an enhanced product description:"""
                                    try:
                                        response = model.generate_content(prompt)
                                        st.markdown(response.text)
                                    except Exception as e:
                                        st.error(f"Error generating description: {e}")
                else:
                    st.info("No product data available to generate descriptions.")

                st.markdown("---")
                st.subheader(f"💬 Chat with Your Data ({DATASET_FILENAME})")
                user_question = st.text_area(f"Ask a question about the {DATASET_FILENAME}:", height=100, key="ai_question")
                if st.button("💬 Get Answer from AI", key="get_answer_btn"):
                    if user_question:
                        with st.spinner("Thinking..."):
                            data_summary = f"""The dataset is '{DATASET_FILENAME}'. It has the following columns: {df.columns.tolist()}.
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

    with tab3:
        st.header("🔬 Advanced Analytics Toolkit")
        st.write("Explore a range of advanced analytical techniques. Select a category to see available tools.")
        st.caption(f"Note: Most tools listed here are conceptual placeholders or require specific column types. Adaptability to '{DATASET_FILENAME}' will vary. Ensure your selected columns are appropriate for each analysis.")

        # Category 1: Advanced Statistical Modeling (ASM)
        with st.expander("📈 Advanced Statistical Modeling (ASM)"):
            st.write("Implementations of advanced statistical models. Ensure your dataset has appropriate columns for each analysis.")

            # --- Helper functions for column selection ---
            # get_numeric_columns and get_categorical_columns are now defined globally

            numeric_cols_adv = get_numeric_columns(df) 
            categorical_cols_adv = get_categorical_columns(df)
            # date_cols is already defined globally and populated

            # --- ASM 1: Advanced Hypothesis Testing ---
            st.subheader("ASM 1: Advanced Hypothesis Testing")
            test_type = st.selectbox("Select Hypothesis Test", ["Chi-Squared Test", "ANOVA", "Kruskal-Wallis Test"], key="asm_ht_type")

            if test_type == "Chi-Squared Test":
                st.markdown("Tests for independence between two categorical variables.")
                if len(categorical_cols_adv) >= 2:
                    cat_col1 = st.selectbox("Select first categorical variable:", categorical_cols_adv, key="asm_chi_cat1")
                    cat_col2 = st.selectbox("Select second categorical variable:", [c for c in categorical_cols_adv if c != cat_col1], key="asm_chi_cat2")
                    if cat_col1 and cat_col2 and st.button("Run Chi-Squared Test", key="asm_chi_run"):
                        try:
                            contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
                            chi2, p, dof, expected = chi2_contingency(contingency_table)
                            st.write("Contingency Table:")
                            st.dataframe(contingency_table)
                            st.write(f"Chi-Squared Statistic: {chi2:.4f}")
                            st.write(f"P-value: {p:.4f}")
                            st.write(f"Degrees of Freedom: {dof}")
                            if p < 0.05:
                                st.success(f"The p-value ({p:.4f}) is less than 0.05, suggesting a significant association between {cat_col1} and {cat_col2}.")
                            else:
                                st.info(f"The p-value ({p:.4f}) is greater than or equal to 0.05, suggesting no significant association between {cat_col1} and {cat_col2}.")
                        except Exception as e:
                            st.error(f"Error running Chi-Squared test: {e}")
                else:
                    st.warning("Not enough categorical columns (at least 2) available for Chi-Squared test.")

            elif test_type == "ANOVA":
                st.markdown("Compares means of a numerical variable across groups of a categorical variable (assumes normality and equal variances).")
                if categorical_cols_adv and numeric_cols_adv:
                    cat_col_anova = st.selectbox("Select categorical grouping variable:", categorical_cols_adv, key="asm_anova_cat")
                    num_col_anova = st.selectbox("Select numerical variable:", numeric_cols_adv, key="asm_anova_num")
                    if cat_col_anova and num_col_anova and st.button("Run ANOVA", key="asm_anova_run"):
                        try:
                            # Filter out groups with less than 2 samples as f_oneway requires at least 2 samples per group
                            valid_groups = []
                            for group_val in df[cat_col_anova].unique():
                                group_data = df[num_col_anova][df[cat_col_anova] == group_val].dropna()
                                if len(group_data) >= 2:
                                    valid_groups.append(group_data)
                            
                            if len(valid_groups) < 2:
                                st.warning("Need at least two groups with at least two samples each for ANOVA.")
                            else:
                                f_stat, p_val = f_oneway(*valid_groups)
                                st.write(f"F-Statistic: {f_stat:.4f}")
                                st.write(f"P-value: {p_val:.4f}")
                                if p_val < 0.05:
                                    st.success(f"The p-value ({p_val:.4f}) is less than 0.05, suggesting significant differences in the mean of {num_col_anova} across groups of {cat_col_anova}.")
                                else:
                                    st.info(f"The p-value ({p_val:.4f}) is greater than or equal to 0.05, suggesting no significant differences in the mean of {num_col_anova} across groups of {cat_col_anova}.")
                        except Exception as e:
                            st.error(f"Error running ANOVA: {e}")
                else:
                    st.warning("ANOVA requires at least one categorical and one numerical column.")

            elif test_type == "Kruskal-Wallis Test":
                st.markdown("Non-parametric alternative to ANOVA. Compares medians of a numerical variable across groups of a categorical variable.")
                if categorical_cols_adv and numeric_cols_adv:
                    cat_col_kw = st.selectbox("Select categorical grouping variable:", categorical_cols_adv, key="asm_kw_cat")
                    num_col_kw = st.selectbox("Select numerical variable:", numeric_cols_adv, key="asm_kw_num")
                    if cat_col_kw and num_col_kw and st.button("Run Kruskal-Wallis Test", key="asm_kw_run"):
                        try:
                            groups = [df[num_col_kw][df[cat_col_kw] == group].dropna() for group in df[cat_col_kw].unique()]
                            groups = [g for g in groups if len(g) > 0] # Ensure groups are not empty
                            if len(groups) < 2:
                                st.warning("Need at least two groups with data for Kruskal-Wallis test.")
                            else:
                                h_stat, p_val = kruskal(*groups)
                                st.write(f"H-Statistic: {h_stat:.4f}")
                                st.write(f"P-value: {p_val:.4f}")
                                if p_val < 0.05:
                                    st.success(f"The p-value ({p_val:.4f}) is less than 0.05, suggesting significant differences in the distribution of {num_col_kw} across groups of {cat_col_kw}.")
                                else:
                                    st.info(f"The p-value ({p_val:.4f}) is greater than or equal to 0.05, suggesting no significant differences in the distribution of {num_col_kw} across groups of {cat_col_kw}.")
                        except Exception as e:
                            st.error(f"Error running Kruskal-Wallis test: {e}")
                else:
                    st.warning("Kruskal-Wallis test requires at least one categorical and one numerical column.")

            st.markdown("---")
            # --- ASM 2: Logistic Regression ---
            st.subheader("ASM 2: Logistic Regression")
            st.markdown("Model binary outcomes (e.g., predict 'B2B' status, or high/low 'Amount') based on other features.")
            if numeric_cols_adv or categorical_cols_adv:
                st.write("Define Target Variable (Binary):")
                target_source_col_logreg = st.selectbox("Select column to create binary target from (e.g., 'B2B', or 'Amount' to binarize):", [None] + df.columns.tolist(), key="asm_logreg_target_src")
                
                target_col_name_logreg = "logistic_target"
                df_logreg = df.copy()
                target_created_logreg = False

                if target_source_col_logreg:
                    if target_source_col_logreg in df_logreg.columns and df_logreg[target_source_col_logreg].dtype == 'bool':
                        df_logreg[target_col_name_logreg] = df_logreg[target_source_col_logreg].astype(int)
                        st.write(f"Using boolean column '{target_source_col_logreg}' as target (True=1, False=0).")
                        target_created_logreg = True
                    elif target_source_col_logreg in numeric_cols_adv:
                        threshold_default_logreg = df_logreg[target_source_col_logreg].median() if not df_logreg[target_source_col_logreg].empty else 0
                        threshold_logreg = st.number_input(f"Enter threshold for '{target_source_col_logreg}' to define 1 (e.g., > threshold)", value=threshold_default_logreg, key="asm_logreg_thresh")
                        df_logreg[target_col_name_logreg] = (df_logreg[target_source_col_logreg] > threshold_logreg).astype(int)
                        target_created_logreg = True
                    elif target_source_col_logreg in categorical_cols_adv: # Check if it's in the pre-filtered categorical list
                        positive_class_logreg = st.selectbox(f"Select the 'positive' class (1) for '{target_source_col_logreg}':", df_logreg[target_source_col_logreg].unique(), key="asm_logreg_pos_class")
                        if positive_class_logreg is not None:
                            df_logreg[target_col_name_logreg] = (df_logreg[target_source_col_logreg] == positive_class_logreg).astype(int)
                            target_created_logreg = True
                        else:
                            st.warning("Please select a positive class for the categorical target.")
                    else: # Fallback for other object types that might not be in categorical_cols_adv due to nunique
                        if df_logreg[target_source_col_logreg].nunique() == 2:
                             positive_class_logreg = st.selectbox(f"Select the 'positive' class (1) for '{target_source_col_logreg}' (has 2 unique values):", df_logreg[target_source_col_logreg].unique(), key="asm_logreg_pos_class_obj")
                             if positive_class_logreg is not None:
                                 df_logreg[target_col_name_logreg] = (df_logreg[target_source_col_logreg] == positive_class_logreg).astype(int)
                                 target_created_logreg = True
                        else:
                            st.warning(f"Selected column '{target_source_col_logreg}' is not easily convertible to a binary target. Choose a boolean, numeric, or low-cardinality categorical column.")

                    if target_created_logreg:
                        st.write(f"Target variable '{target_col_name_logreg}' created. Value counts:")
                        st.write(df_logreg[target_col_name_logreg].value_counts())

                        feature_cols_options_logreg = [col for col in df.columns if col != target_source_col_logreg and col != target_col_name_logreg]
                        selected_features_logreg = st.multiselect("Select feature variables:", feature_cols_options_logreg, key="asm_logreg_features")

                        if selected_features_logreg and st.button("Run Logistic Regression", key="asm_logreg_run"):
                            try:
                                X = df_logreg[selected_features_logreg].copy()
                                y = df_logreg[target_col_name_logreg].copy()

                                # Handle potential NaN values before get_dummies and scaling/fitting
                                for col in X.select_dtypes(include=np.number).columns:
                                    X[col] = X[col].fillna(X[col].median()) # Impute numeric with median
                                for col in X.select_dtypes(include='object').columns:
                                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown') # Impute categorical with mode

                                X = pd.get_dummies(X, drop_first=True, dummy_na=False) 
                                X = sm.add_constant(X)

                                if X.empty or y.empty or len(X) != len(y):
                                    st.error("Feature set or target variable is empty or mismatched after preprocessing.")
                                elif y.nunique() < 2:
                                    st.error("Target variable must have at least two unique classes for logistic regression.")
                                else:
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None)
                                    
                                    log_reg_model = sm.Logit(y_train, X_train.astype(float)).fit(disp=0) 
                                    st.subheader("Logistic Regression Results")
                                    st.text(log_reg_model.summary())

                                    y_pred_proba = log_reg_model.predict(X_test.astype(float))
                                    y_pred = (y_pred_proba > 0.5).astype(int)

                                    st.subheader("Model Evaluation (Test Set)")
                                    st.text(classification_report(y_test, y_pred, zero_division=0))
                                    cm = confusion_matrix(y_test, y_pred)
                                    fig, ax = plt.subplots()
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                    ax.set_xlabel('Predicted')
                                    ax.set_ylabel('Actual')
                                    ax.set_title('Confusion Matrix')
                                    st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error running Logistic Regression: {e}")
            else:
                st.warning("Logistic regression requires numeric or categorical columns for features and target creation.")

            st.markdown("---")
            # --- ASM 3: Time Series Decomposition & Analysis ---
            st.subheader("ASM 3: Time Series Decomposition & Analysis")
            if not date_cols: # Global date_cols
                st.warning("No datetime columns found in the dataset. Time series analysis requires a date column.")
            elif not numeric_cols_adv:
                st.warning("No numeric columns found for time series values.")
            else:
                time_col_tsa = st.selectbox("Select Date/Time column:", date_cols, index=date_cols.index('Date') if 'Date' in date_cols else 0, key="asm_ts_datecol")
                value_col_tsa = st.selectbox("Select Value column for time series:", numeric_cols_adv, index=numeric_cols_adv.index('Amount') if 'Amount' in numeric_cols_adv else 0, key="asm_ts_valcol")
                aggregation_freq_tsa = st.selectbox("Aggregate data by:", ["D", "W", "M", "Q", "Y"], index=2, key="asm_ts_freq", help="D: Day, W: Week, M: Month, Q: Quarter, Y: Year")
                
                if time_col_tsa and value_col_tsa and st.button("Analyze Time Series", key="asm_ts_run"):
                    try:
                        ts_df = df[[time_col_tsa, value_col_tsa]].copy()
                        ts_df[time_col_tsa] = pd.to_datetime(ts_df[time_col_tsa], errors='coerce')
                        ts_df = ts_df.dropna(subset=[time_col_tsa, value_col_tsa])
                        ts_df = ts_df.set_index(time_col_tsa)
                        
                        ts_aggregated = ts_df[value_col_tsa].resample(aggregation_freq_tsa).sum() 
                        ts_aggregated = ts_aggregated.dropna()

                        min_periods_decompose_tsa = 12 
                        min_periods_arima_tsa = 20    

                        if len(ts_aggregated) < 4 : # Absolute minimum for any decomposition/ARIMA
                             st.warning(f"Not enough data points ({len(ts_aggregated)}) after aggregation for Time Series Analysis. Need at least 4. For meaningful results, aim for {min_periods_decompose_tsa}+ for decomposition and {min_periods_arima_tsa}+ for ARIMA.")
                        else:
                            st.write(f"Aggregated Time Series (first 5 rows, {aggregation_freq_tsa} frequency, sum of {value_col_tsa}):")
                            st.write(ts_aggregated.head())

                            if len(ts_aggregated) >= min_periods_decompose_tsa:
                                st.subheader("Time Series Decomposition")
                                period_map_tsa = {'D': 7, 'W': 4, 'M': 12, 'Q': 4, 'Y':1} 
                                decomp_period_tsa = period_map_tsa.get(aggregation_freq_tsa, 1)
                                # Ensure period is not too large for the data, and at least 2 for seasonal_decompose
                                decomp_period_tsa = max(2, min(decomp_period_tsa, len(ts_aggregated) // 2)) if len(ts_aggregated) >=4 else 1
                                
                                if decomp_period_tsa > 1:
                                    decomposition = seasonal_decompose(ts_aggregated, model='additive', period=decomp_period_tsa ) 
                                    fig_decompose = decomposition.plot()
                                    fig_decompose.set_size_inches(10, 8)
                                    st.pyplot(fig_decompose)
                                else:
                                    st.info("Not enough data or period is too small for seasonal decomposition. Plotting raw series.")
                                    fig_raw, ax_raw = plt.subplots()
                                    ts_aggregated.plot(ax=ax_raw)
                                    ax_raw.set_title("Aggregated Time Series")
                                    st.pyplot(fig_raw)
                            else:
                                st.warning(f"Not enough data points ({len(ts_aggregated)}) for meaningful decomposition. Need at least ~{min_periods_decompose_tsa} for the chosen aggregation '{aggregation_freq_tsa}'.")


                            st.subheader("ARIMA Model (Example)")
                            if len(ts_aggregated) >= min_periods_arima_tsa:
                                try:
                                    # Simpler default order, or consider auto_arima if library is added
                                    arima_order = (1,1,1) if len(ts_aggregated) > 20 else (1,0,0) 
                                    model_arima = ARIMA(ts_aggregated, order=arima_order).fit() 
                                    st.text(model_arima.summary())
                                    
                                    forecast_steps_arima = min(12, max(1, len(ts_aggregated)//4))
                                    forecast_arima = model_arima.get_forecast(steps=forecast_steps_arima)
                                    forecast_df_arima = forecast_arima.summary_frame()

                                    fig_arima, ax_arima = plt.subplots(figsize=(10, 6))
                                    ts_aggregated.plot(ax=ax_arima, label='Observed')
                                    forecast_df_arima['mean'].plot(ax=ax_arima, label='Forecast')
                                    ax_arima.fill_between(forecast_df_arima.index, forecast_df_arima['mean_ci_lower'], forecast_df_arima['mean_ci_upper'], color='k', alpha=.15)
                                    ax_arima.set_title(f'ARIMA Forecast for {value_col_tsa}')
                                    ax_arima.legend()
                                    st.pyplot(fig_arima)
                                except Exception as e_arima:
                                    st.error(f"Error fitting/forecasting with ARIMA: {e_arima}. Try different ARIMA orders, ensure stationary data, or check data length.")
                            else:
                                st.warning(f"Not enough data points ({len(ts_aggregated)}) for a reliable ARIMA model example after aggregation. Need at least ~{min_periods_arima_tsa}.")
                    except Exception as e:
                        st.error(f"Error in Time Series Analysis: {e}")

            st.markdown("---")
            # --- ASM 4: Survival Analysis (Conceptual Example) ---
            st.subheader("ASM 4: Survival Analysis (Conceptual Example)")
            st.markdown("""
            Survival analysis studies the time until an event occurs. 
            The current dataset might not have direct duration/event columns (e.g., time-to-repurchase, subscription length).
            This is a conceptual demonstration using synthetically generated data.
            """)
            if st.button("Show Survival Analysis Example", key="asm_sa_run"):
                try:
                    np.random.seed(42)
                    N_sa = 200
                    T_sa = np.random.exponential(scale=10, size=N_sa) + np.random.normal(loc=5, scale=2, size=N_sa)
                    T_sa = np.clip(T_sa, 1, 50) 
                    E_sa = np.random.binomial(1, 0.7, size=N_sa) 
                    
                    censoring_time_sa = np.random.uniform(5, 40, size=N_sa)
                    T_sa[E_sa==0] = np.minimum(T_sa[E_sa==0], censoring_time_sa[E_sa==0])

                    st.write("Synthetic Data Sample (First 10 entries):")
                    st.write(pd.DataFrame({'Duration (T)': T_sa, 'EventOccurred (E)': E_sa}).head(10))

                    kmf = KaplanMeierFitter()
                    kmf.fit(T_sa, event_observed=E_sa)

                    fig_km, ax_km = plt.subplots(figsize=(8,6))
                    kmf.plot_survival_function(ax=ax_km)
                    ax_km.set_title('Kaplan-Meier Survival Curve (Synthetic Data)')
                    ax_km.set_xlabel('Time')
                    ax_km.set_ylabel('Survival Probability')
                    st.pyplot(fig_km)

                    st.write("Median Survival Time (where survival probability is 0.5):")
                    st.write(f"{kmf.median_survival_time_:.2f} time units")
                except Exception as e:
                    st.error(f"Error in Survival Analysis example: {e}")

            st.markdown("---")
            # --- ASM 5: Bayesian A/B Testing ---
            st.subheader("ASM 5: Bayesian A/B Testing")
            st.markdown("Compare two groups (e.g., from 'Sales Channel' or 'Fulfilment') on a binary outcome (e.g., 'B2B' status) using Bayesian methods.")
            
            group_col_ab_options_adv = [None] + categorical_cols_adv 
            conversion_col_ab_options_adv = [None] + df.columns.tolist() 

            group_col_ab_adv = st.selectbox("Select Grouping Column (Categorical with 2 distinct values):", group_col_ab_options_adv, key="asm_bab_group")
            conversion_col_ab_adv_src = st.selectbox("Select Binary Outcome Column (e.g., 'B2B', or a 0/1 column):", conversion_col_ab_options_adv, key="asm_bab_conversion_src")

            if group_col_ab_adv and conversion_col_ab_adv_src:
                if df[group_col_ab_adv].nunique() != 2:
                    st.warning(f"Selected Grouping Column '{group_col_ab_adv}' must have exactly two unique values for this A/B test setup. It has {df[group_col_ab_adv].nunique()}.")
                elif st.button("Run Bayesian A/B Test", key="asm_bab_run"):
                    try:
                        temp_df_bab = df[[group_col_ab_adv, conversion_col_ab_adv_src]].copy().dropna()
                        conversion_col_bab_final = "bab_conversion_metric"
                        
                        # Ensure conversion column is binary 0/1
                        if pd.api.types.is_bool_dtype(temp_df_bab[conversion_col_ab_adv_src]):
                            temp_df_bab[conversion_col_bab_final] = temp_df_bab[conversion_col_ab_adv_src].astype(int)
                        elif pd.api.types.is_numeric_dtype(temp_df_bab[conversion_col_ab_adv_src]) and temp_df_bab[conversion_col_ab_adv_src].isin([0,1]).all():
                             temp_df_bab[conversion_col_bab_final] = temp_df_bab[conversion_col_ab_adv_src]
                        else: # Attempt to binarize if not already binary
                            st.info(f"Outcome column '{conversion_col_ab_adv_src}' is not directly binary. Trying to binarize.")
                            if pd.api.types.is_numeric_dtype(temp_df_bab[conversion_col_ab_adv_src]):
                                bab_thresh = st.number_input(f"Enter threshold for '{conversion_col_ab_adv_src}' to make it binary for B.A/B:", value=temp_df_bab[conversion_col_ab_adv_src].median(), key="bab_thresh_auto")
                                temp_df_bab[conversion_col_bab_final] = (temp_df_bab[conversion_col_ab_adv_src] > bab_thresh).astype(int)
                            elif temp_df_bab[conversion_col_ab_adv_src].nunique() == 2: # Categorical with 2 values
                                positive_class_bab = st.selectbox(f"Select positive class for '{conversion_col_ab_adv_src}' for B.A/B:", temp_df_bab[conversion_col_ab_adv_src].unique(), key="bab_pos_class_auto")
                                temp_df_bab[conversion_col_bab_final] = (temp_df_bab[conversion_col_ab_adv_src] == positive_class_bab).astype(int)
                            else:
                                st.error(f"The Outcome Column '{conversion_col_ab_adv_src}' must be binary (0 or 1, or True/False) or binarizable. Please preprocess or select an appropriate column.")
                                st.stop()
                        
                        if not temp_df_bab[conversion_col_bab_final].isin([0,1]).all():
                            st.error(f"Failed to create a binary outcome column '{conversion_col_bab_final}'. Check source column and binarization logic.")
                            st.stop()

                        summary_bab = temp_df_bab.groupby(group_col_ab_adv)[conversion_col_bab_final].agg(['sum', 'count'])
                        summary_bab.columns = ['conversions', 'total_users']
                        st.write("A/B Test Data Summary:")
                        st.write(summary_bab)

                        if len(summary_bab) == 2: 
                            group_names_bab = summary_bab.index.tolist()
                            conversions_a_bab = summary_bab.loc[group_names_bab[0], 'conversions']
                            total_a_bab = summary_bab.loc[group_names_bab[0], 'total_users']
                            conversions_b_bab = summary_bab.loc[group_names_bab[1], 'conversions']
                            total_b_bab = summary_bab.loc[group_names_bab[1], 'total_users']

                            with pm.Model() as bayesian_ab_model:
                                p_A = pm.Beta(f'p_{group_names_bab[0]}', alpha=1, beta=1)
                                p_B = pm.Beta(f'p_{group_names_bab[1]}', alpha=1, beta=1)
                                obs_A = pm.Binomial(f'obs_{group_names_bab[0]}', n=total_a_bab, p=p_A, observed=conversions_a_bab)
                                obs_B = pm.Binomial(f'obs_{group_names_bab[1]}', n=total_b_bab, p=p_B, observed=conversions_b_bab)
                                delta = pm.Deterministic('delta', p_B - p_A)
                                uplift_abs = pm.Deterministic('uplift_abs', p_B - p_A) # Same as delta, but for clarity
                                uplift_rel = pm.Deterministic('uplift_rel', (p_B - p_A) / p_A if p_A > 0 else 0) 
                                
                                # Use return_inferencedata=True for ArviZ compatibility
                                trace = pm.sample(2000, tune=1000, cores=1, progressbar=True, return_inferencedata=True)

                            st.subheader("Bayesian A/B Test Results")
                            
                            fig_posterior, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=False) # sharex=False might be better for different scales
                            az.plot_posterior(trace, var_names=[f'p_{group_names_bab[0]}'], ax=axes[0], show=False, hdi_prob=0.95)
                            az.plot_posterior(trace, var_names=[f'p_{group_names_bab[1]}'], ax=axes[1], show=False, hdi_prob=0.95)
                            az.plot_posterior(trace, var_names=['delta'], ax=axes[2], show=False, hdi_prob=0.95, ref_val=0)
                            az.plot_posterior(trace, var_names=['uplift_rel'], ax=axes[3], show=False, hdi_prob=0.95, ref_val=0)
                            
                            axes[0].set_title(f'Posterior of rate for {group_names_bab[0]}')
                            axes[1].set_title(f'Posterior of rate for {group_names_bab[1]}')
                            axes[2].set_title(f'Posterior of absolute difference ({group_names_bab[1]} - {group_names_bab[0]})')
                            axes[3].set_title(f'Posterior of relative uplift ({group_names_bab[1]} vs {group_names_bab[0]})')
                            plt.tight_layout()
                            st.pyplot(fig_posterior)

                            prob_b_better_a_bab = (trace.posterior['delta'].values > 0).mean()
                            st.write(f"Probability that Group '{group_names_bab[1]}'s rate is greater than Group '{group_names_bab[0]}'s: {prob_b_better_a_bab:.2%}")

                            hdi_delta = az.hdi(trace.posterior['delta'], hdi_prob=0.95).x.values
                            st.write(f"95% Highest Density Interval for delta (difference): [{hdi_delta[0]:.4f}, {hdi_delta[1]:.4f}]")

                            if prob_b_better_a_bab > 0.95 and hdi_delta[0] > 0: # Strong evidence B is better
                                st.success(f"Strong evidence that Group '{group_names_bab[1]}' performs better than Group '{group_names_bab[0]}'.")
                            elif prob_b_better_a_bab < 0.05 and hdi_delta[1] < 0: # Strong evidence A is better
                                st.success(f"Strong evidence that Group '{group_names_bab[0]}' performs better than Group '{group_names_bab[1]}'.")
                            else:
                                st.info("The evidence is not strong enough to definitively conclude one group is better than the other based on a 95% HDI and probability threshold.")
                        else:
                            st.warning("Bayesian A/B testing requires exactly two distinct groups in the selected 'Grouping Column'.")
                    except Exception as e:
                        st.error(f"Error running Bayesian A/B Test: {e}")
            else:
                st.warning(f"Bayesian A/B testing requires a 'Grouping Column' (categorical with 2 levels) and a binary 'Outcome Column' (e.g., 'B2B' status, 0/1).")
        
        # Category 2: Machine Learning - Supervised (MLS)
        with st.expander("🤖 Machine Learning - Supervised (MLS)"):
            # Content for Supervised Machine Learning models will go here.
            # The RFM analysis code previously here was a duplicate from Tab 1
            # and caused an IndentationError. It has been removed.
            pass # Placeholder for actual MLS content
        # The following analysis blocks (Customer Churn Detection, Sales Forecasting,

except FileNotFoundError:
    st.error(f"🚨 Error: `{DATASET_FILENAME}` not found. Please make sure the file is in the same directory as `app.py`.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error(f"🚨 Error: `{DATASET_FILENAME}` is empty. Please provide a valid CSV file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading or initial setup: {e}")
    st.stop()

