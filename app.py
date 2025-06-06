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
from sklearn.ensemble import RandomForestClassifier # Added for new MLS tool
from sklearn.ensemble import RandomForestRegressor # For new ML regression tool
from sklearn.metrics import mean_squared_error, r2_score # For new ML regression tool
from sklearn.decomposition import PCA # For new PCA tool
# import plotly.express as px # For advanced interactive visualizations
from lifelines import KaplanMeierFitter
import pymc as pm
import arviz as az
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
# import shap # For model interpretability
# from gensim.models import LdaModel # For topic modeling
# # import tensorflow as tf # For deep learning (conceptual)
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="🛒 E-commerce Sales Insights Suite", layout="wide")
st.title("🛒 E-commerce Sales Insights Suite")
st.write("Exploring insights from the preloaded Amazon Sales Report. Analyze segments, sales trends, and more!")

# --- Custom CSS for Scrollable Tabs ---
st.markdown("""
<style>
    /* Target the container for the tab buttons */
    div[data-baseweb="tabs"] > div:first-child > div:first-child {
        overflow-x: auto !important; /* Enable horizontal scroll */
        white-space: nowrap !important; /* Prevent tabs from wrapping to the next line */
        scrollbar-width: thin; /* Optional: For a thinner scrollbar in Firefox */
        scrollbar-color: #888 #f0f0f0; /* Optional: Scrollbar colors for Firefox */
    }
    /* For Webkit browsers like Chrome, Safari */
    div[data-baseweb="tabs"] > div:first-child > div:first-child::-webkit-scrollbar {
        height: 8px; /* Height of the scrollbar */
    }
    div[data-baseweb="tabs"] > div:first-child > div:first-child::-webkit-scrollbar-thumb {
        background-color: #888; /* Color of the scrollbar thumb */
        border-radius: 4px; /* Rounded corners for the thumb */
    }
    div[data-baseweb="tabs"] > div:first-child > div:first-child::-webkit-scrollbar-track {
        background-color: #f0f0f0; /* Color of the scrollbar track */
    }
    /* Ensure individual tab buttons don't shrink excessively, allowing them to overflow */
    div[data-baseweb="tabs"] > div:first-child > div:first-child > button {
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Trad. Analysis",
        "🤖 AI Insights",
        "🔬 Adv. Analytics",
        "🔍 Data QA/EDA",
        "🔧 Feature Eng.",
        "🧠 Modeling Lab",
        "📈 Visuals/Export"
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
        with st.expander("📈 Overall Promotion Impact Analysis (Time-Based)", expanded=False):
            st.info("Analyze the overall impact of promotions on sales metrics over time. Compare trends for promoted vs. non-promoted sales and identify top-performing promotion IDs. This tool relies on a 'promotion-ids' column.")

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
                order_id_col_promo = st.selectbox("Select Order ID column (for AOV):", all_cols_promo, index=all_cols_promo.index('Order ID') if 'Order ID' in all_cols_promo else 0, key="promo_order_id")

            st.markdown("#### Filters")
            min_date_promo = df[date_col_promo].min() if date_col_promo and date_col_promo in df.columns and not df[date_col_promo].isnull().all() else datetime.now().date() - pd.Timedelta(days=365)
            max_date_promo = df[date_col_promo].max() if date_col_promo and date_col_promo in df.columns and not df[date_col_promo].isnull().all() else datetime.now().date()
            if isinstance(min_date_promo, pd.Timestamp): min_date_promo = min_date_promo.date()
            if isinstance(max_date_promo, pd.Timestamp): max_date_promo = max_date_promo.date()

            start_date_promo = st.date_input("Start date for Promo Analysis:", min_date_promo, min_value=min_date_promo, max_value=max_date_promo, key="promo_start_date")
            end_date_promo = st.date_input("End date for Promo Analysis:", max_date_promo, min_value=min_date_promo, max_value=max_date_promo, key="promo_end_date")

            if start_date_promo > end_date_promo:
                st.warning("Start date cannot be after end date for Promotion Insights.")
            
            aggregation_freq_promo = st.selectbox("Aggregate trend by:", ["D", "W", "M"], index=2, format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x], key="promo_agg_freq")

            if st.button("💡 Run Overall Promotion Impact Analysis", key="promo_run"):
                if not all([date_col_promo, amount_col_promo, qty_col_promo, promo_ids_col, order_id_col_promo]):
                    st.warning("Please select Date, Amount, Quantity, Promotion IDs, and Order ID columns.")
                elif start_date_promo > end_date_promo:
                    st.warning("Correct the date range before running analysis.")
                else:
                    try:
                        promo_df = df.copy()
                        promo_df[date_col_promo] = pd.to_datetime(promo_df[date_col_promo], errors='coerce')
                        promo_df = promo_df.dropna(subset=[date_col_promo, amount_col_promo, qty_col_promo, order_id_col_promo]) # Promo ID can be NaN
                        promo_df = promo_df[(promo_df[date_col_promo] >= pd.to_datetime(start_date_promo)) & (promo_df[date_col_promo] <= pd.to_datetime(end_date_promo))]

                        if promo_df.empty:
                            st.warning("No data available for the selected criteria in Promotion Insights.")
                        else:
                            st.subheader("Promotion Insights Results")

                            # Define 'HasPromotion' based on whether promo_ids_col is NaN or empty
                            promo_df['HasPromotion'] = ~promo_df[promo_ids_col].isnull() & (promo_df[promo_ids_col].astype(str).str.strip() != '')

                            st.markdown("#### Overall Promotion Status Counts (Line Items)")
                            promo_summary = promo_df['HasPromotion'].value_counts().rename(index={True: 'With Promotion', False: 'Without Promotion'})
                            st.bar_chart(promo_summary)
                            st.write(promo_summary)

                            if promo_df['HasPromotion'].any(): # If there are any promoted items
                                # Time-based analysis
                                promo_df['TimePeriod'] = promo_df[date_col_promo].dt.to_period(aggregation_freq_promo)
                                
                                # Revenue Trend
                                revenue_trend = promo_df.groupby(['TimePeriod', 'HasPromotion'])[amount_col_promo].sum().unstack(fill_value=0)
                                revenue_trend.columns = ['Revenue_NonPromoted', 'Revenue_Promoted'] if False in revenue_trend.columns and True in revenue_trend.columns else (['Revenue_Promoted'] if True in revenue_trend.columns else ['Revenue_NonPromoted']) # Handle cases where one type is missing
                                st.markdown(f"###### Total Revenue Trend ({aggregation_freq_promo})")
                                st.line_chart(revenue_trend)

                                # AOV Trend (for orders with any promoted item vs. orders with no promoted items)
                                order_promo_status_time = promo_df.groupby([order_id_col_promo, 'TimePeriod'])['HasPromotion'].any().reset_index(name='OrderHasPromotion')
                                order_data_time = promo_df.groupby([order_id_col_promo, 'TimePeriod'])[amount_col_promo].sum().reset_index(name='OrderAmount')
                                order_data_time = pd.merge(order_data_time, order_promo_status_time, on=[order_id_col_promo, 'TimePeriod'])
                                
                                aov_trend_data = order_data_time.groupby(['TimePeriod', 'OrderHasPromotion'])['OrderAmount'].agg(['sum', 'count'])
                                aov_trend_data['AOV'] = aov_trend_data['sum'] / aov_trend_data['count']
                                aov_trend_plot = aov_trend_data['AOV'].unstack(fill_value=0)
                                aov_trend_plot.columns = ['AOV_NonPromotedOrder', 'AOV_PromotedOrder'] if False in aov_trend_plot.columns and True in aov_trend_plot.columns else (['AOV_PromotedOrder'] if True in aov_trend_plot.columns else ['AOV_NonPromotedOrder'])
                                st.markdown(f"###### Average Order Value (AOV) Trend ({aggregation_freq_promo})")
                                st.line_chart(aov_trend_plot)

                                # Overall Summary Table
                                st.markdown("###### Overall Performance Summary (Promoted vs. Non-Promoted Line Items)")
                                overall_item_summary = promo_df.groupby('HasPromotion').agg(
                                    TotalRevenue=(amount_col_promo, 'sum'),
                                    TotalQuantity=(qty_col_promo, 'sum'),
                                    AverageItemPrice=(amount_col_promo, 'mean'),
                                    NumberOfLineItems=(order_id_col_promo, 'count') 
                                ).rename(index={True: 'With Promotion', False: 'Without Promotion'})
                                st.dataframe(overall_item_summary)

                                # Overall AOV for orders with promoted items vs. no promoted items
                                overall_order_promo_status = promo_df.groupby(order_id_col_promo)['HasPromotion'].any().reset_index(name='OrderHasPromotion')
                                overall_order_data = promo_df.groupby(order_id_col_promo)[amount_col_promo].sum().reset_index(name='OrderAmount')
                                overall_order_data = pd.merge(overall_order_data, overall_order_promo_status, on=order_id_col_promo)
                                overall_aov_summary = overall_order_data.groupby('OrderHasPromotion')['OrderAmount'].agg(['sum', 'count', 'mean'])
                                overall_aov_summary.columns = ['TotalRevenue', 'NumberOfOrders', 'AverageOrderValue']
                                overall_aov_summary = overall_aov_summary.rename(index={True: 'Orders with Promoted Item(s)', False: 'Orders with No Promoted Items'})
                                st.markdown("###### Overall Order Performance (Orders with vs. without Promoted Items)")
                                st.dataframe(overall_aov_summary)

                                # Top Promotion IDs by Revenue (if IDs are meaningful and not too many)
                                if promo_df[promo_df['HasPromotion']][promo_ids_col].nunique() > 0 and promo_df[promo_df['HasPromotion']][promo_ids_col].nunique() < 100: # Avoid too many unique IDs
                                    st.markdown(f"#### Top Promotion IDs by Revenue (from '{promo_ids_col}')")
                                    top_promo_codes = promo_df[promo_df['HasPromotion']].groupby(promo_ids_col)[amount_col_promo].sum().nlargest(10)
                                    if not top_promo_codes.empty:
                                        st.write("Top 10 Promotion IDs by Total Revenue Generated:")
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

        # New Tool: Sales Breakdown by Attribute
        with st.expander("📊 Sales Breakdown by Attribute", expanded=False):
            st.info("Analyze sales metrics (e.g., Amount, Quantity) broken down by a selected categorical attribute. Allows for flexible exploration of performance across different dimensions.")

            all_cols_sba = df.columns.tolist()
            # Using globally defined get_categorical_columns, get_numeric_columns, date_cols

            st.markdown("#### Column Selection")
            attribute_col_sba = st.selectbox(
                "Select Attribute Column (Categorical):",
                [None] + get_categorical_columns(df, nunique_threshold=50), # Allow more unique values
                index=0,
                key="sba_attribute"
            )
            metric_col_sba = st.selectbox(
                "Select Metric Column (Numeric - e.g., Amount, Qty):",
                [None] + get_numeric_columns(df),
                index=0,
                key="sba_metric"
            )
            aggregation_sba = st.selectbox(
                "Select Aggregation Function:",
                ['Sum', 'Mean', 'Count'],
                index=0,
                key="sba_aggregation"
            )

            order_id_col_sba_count = None
            if aggregation_sba == 'Count':
                order_id_col_sba_count = st.selectbox(
                    "Select Order ID column (for counting unique orders, optional):",
                    [None] + all_cols_sba,
                    index=([None] + all_cols_sba).index('Order ID') if 'Order ID' in all_cols_sba else 0,
                    key="sba_order_id_for_count",
                    help="If selected, 'Count' will represent unique orders. Otherwise, it counts line items/rows."
                )

            st.markdown("#### Filters & Options")
            top_n_sba = st.slider("Select Top N attribute values to display in chart:", 5, 30, 10, key="sba_top_n")

            date_col_sba_filter = st.selectbox(
                "Optional: Select Date column for filtering:",
                [None] + date_cols, # Global date_cols
                index=0,
                key="sba_date_filter"
            )
            start_date_sba, end_date_sba = None, None
            if date_col_sba_filter and date_col_sba_filter in df.columns and not df[date_col_sba_filter].isnull().all():
                min_date_sba_val = df[date_col_sba_filter].min()
                max_date_sba_val = df[date_col_sba_filter].max()
                if isinstance(min_date_sba_val, pd.Timestamp): min_date_sba_val = min_date_sba_val.date()
                if isinstance(max_date_sba_val, pd.Timestamp): max_date_sba_val = max_date_sba_val.date()

                sba_date_col1, sba_date_col2 = st.columns(2)
                with sba_date_col1:
                    start_date_sba = st.date_input("Start date for filter:", min_date_sba_val, min_value=min_date_sba_val, max_value=max_date_sba_val, key="sba_start_date")
                with sba_date_col2:
                    end_date_sba = st.date_input("End date for filter:", max_date_sba_val, min_value=min_date_sba_val, max_value=max_date_sba_val, key="sba_end_date")
                if start_date_sba and end_date_sba and start_date_sba > end_date_sba:
                    st.warning("Start date cannot be after end date for filtering.")

            if st.button("🔍 Run Sales Breakdown Analysis", key="sba_run"):
                if not attribute_col_sba or not metric_col_sba:
                    st.warning("Please select an Attribute column and a Metric column.")
                elif date_col_sba_filter and start_date_sba and end_date_sba and start_date_sba > end_date_sba:
                    st.warning("Correct the date range before running analysis.")
                else:
                    try:
                        sba_df_filtered = df.copy()
                        if date_col_sba_filter and start_date_sba and end_date_sba:
                            sba_df_filtered[date_col_sba_filter] = pd.to_datetime(sba_df_filtered[date_col_sba_filter], errors='coerce')
                            sba_df_filtered = sba_df_filtered[(sba_df_filtered[date_col_sba_filter] >= pd.to_datetime(start_date_sba)) & (sba_df_filtered[date_col_sba_filter] <= pd.to_datetime(end_date_sba))]

                        sba_df_filtered = sba_df_filtered.dropna(subset=[attribute_col_sba, metric_col_sba])

                        if sba_df_filtered.empty:
                            st.warning("No data available for the selected criteria in Sales Breakdown Analysis.")
                        else:
                            st.subheader(f"Sales Breakdown: {aggregation_sba} of '{metric_col_sba}' by '{attribute_col_sba}'")

                            if aggregation_sba == 'Sum':
                                grouped_data = sba_df_filtered.groupby(attribute_col_sba)[metric_col_sba].sum()
                            elif aggregation_sba == 'Mean':
                                grouped_data = sba_df_filtered.groupby(attribute_col_sba)[metric_col_sba].mean()
                            elif aggregation_sba == 'Count':
                                if order_id_col_sba_count and order_id_col_sba_count in sba_df_filtered.columns:
                                    grouped_data = sba_df_filtered.groupby(attribute_col_sba)[order_id_col_sba_count].nunique()
                                    st.caption(f"Counting unique '{order_id_col_sba_count}' per '{attribute_col_sba}'.")
                                else:
                                    grouped_data = sba_df_filtered.groupby(attribute_col_sba).size()
                                    st.caption(f"Counting rows/items per '{attribute_col_sba}'. Select Order ID for unique order count.")
                            else:
                                st.error("Invalid aggregation function selected."); st.stop()

                            grouped_data = grouped_data.sort_values(ascending=False)

                            if grouped_data.empty:
                                st.info(f"No data to display for '{attribute_col_sba}' breakdown.")
                            else:
                                st.markdown(f"#### Top {top_n_sba} '{attribute_col_sba}' values by {aggregation_sba} of '{metric_col_sba}'")
                                st.bar_chart(grouped_data.head(top_n_sba))
                                st.dataframe(grouped_data.reset_index().head(min(len(grouped_data), 50)))
                    except Exception as e:
                        st.error(f"An error occurred during Sales Breakdown Analysis: {e}")

        # New Tool: Sales by Time Patterns
        with st.expander("🕒 Sales by Time Patterns (Day of Week / Hour of Day)", expanded=False):
            st.info("Analyze sales metrics (e.g., Amount, Quantity, Order Count) based on the day of the week or hour of the day transactions occurred.")

            all_cols_stp = df.columns.tolist()
            # Using globally defined get_numeric_columns, date_cols

            st.markdown("#### Column Selection")
            date_col_stp = st.selectbox(
                "Select Date Column:",
                date_cols, # Global date_cols
                index=date_cols.index('Date') if 'Date' in date_cols else 0,
                key="stp_date"
            )
            metric_col_stp = st.selectbox(
                "Select Metric Column (e.g., Amount, Qty):",
                # Allow None, and ensure numeric_cols is used
                [None] + get_numeric_columns(df),
                index=0, # Default to None
                key="stp_metric"
            )
            aggregation_stp = st.selectbox(
                "Select Aggregation Function:",
                ['Sum', 'Mean', 'Count of Orders'],
                index=0,
                key="stp_aggregation"
            )

            order_id_col_stp_count = None
            if aggregation_stp == 'Count of Orders':
                order_id_col_stp_count = st.selectbox(
                    "Select Order ID column (for counting unique orders):",
                    all_cols_stp,
                    index=all_cols_stp.index('Order ID') if 'Order ID' in all_cols_stp else 0,
                    key="stp_order_id_for_count",
                    help="Required if 'Count of Orders' is selected."
                )

            analysis_type_stp = st.radio(
                "Analyze by:",
                ('Day of Week', 'Hour of Day'),
                key="stp_analysis_type"
            )

            st.markdown("#### Filters & Options")
            start_date_stp, end_date_stp = None, None
            if date_col_stp and date_col_stp in df.columns and not df[date_col_stp].isnull().all():
                min_date_stp_val = df[date_col_stp].min()
                max_date_stp_val = df[date_col_stp].max()
                if isinstance(min_date_stp_val, pd.Timestamp): min_date_stp_val = min_date_stp_val.date()
                if isinstance(max_date_stp_val, pd.Timestamp): max_date_stp_val = max_date_stp_val.date()

                stp_date_col1, stp_date_col2 = st.columns(2)
                with stp_date_col1:
                    start_date_stp = st.date_input("Start date for time pattern filter (optional):", min_date_stp_val, min_value=min_date_stp_val, max_value=max_date_stp_val, key="stp_start_date")
                with stp_date_col2:
                    end_date_stp = st.date_input("End date for time pattern filter (optional):", max_date_stp_val, min_value=min_date_stp_val, max_value=max_date_stp_val, key="stp_end_date")
                if start_date_stp and end_date_stp and start_date_stp > end_date_stp:
                    st.warning("Start date cannot be after end date for filtering.")

            if st.button("🕰️ Run Time Pattern Analysis", key="stp_run"):
                if not date_col_stp or not metric_col_stp: # metric_col_stp is always needed for selection, even if Count of Orders uses order_id_col
                    st.warning("Please select a Date column and a Metric column.")
                elif aggregation_stp == 'Count of Orders' and not order_id_col_stp_count:
                    st.warning("Please select an Order ID column when aggregation is 'Count of Orders'.")
                elif start_date_stp and end_date_stp and start_date_stp > end_date_stp:
                    st.warning("Correct the date range before running analysis.")
                else:
                    try:
                        stp_df_filtered = df.copy()
                        if start_date_stp and end_date_stp:
                            stp_df_filtered[date_col_stp] = pd.to_datetime(stp_df_filtered[date_col_stp], errors='coerce')
                            stp_df_filtered = stp_df_filtered[
                                (stp_df_filtered[date_col_stp] >= pd.to_datetime(start_date_stp)) &
                                (stp_df_filtered[date_col_stp] <= pd.to_datetime(end_date_stp))
                            ]
                        
                        stp_df_filtered[date_col_stp] = pd.to_datetime(stp_df_filtered[date_col_stp], errors='coerce')
                        
                        required_cols_for_na_drop_stp = [date_col_stp]
                        if aggregation_stp == 'Count of Orders':
                            required_cols_for_na_drop_stp.append(order_id_col_stp_count)
                        else: # For Sum or Mean
                            required_cols_for_na_drop_stp.append(metric_col_stp)
                        
                        stp_df_filtered = stp_df_filtered.dropna(subset=required_cols_for_na_drop_stp)

                        if stp_df_filtered.empty:
                            st.warning("No data available for the selected criteria in Time Pattern Analysis.")
                        else:
                            group_by_col_name = ""
                            if analysis_type_stp == 'Day of Week':
                                group_by_col_name = 'DayOfWeek'
                                stp_df_filtered[group_by_col_name] = stp_df_filtered[date_col_stp].dt.day_name()
                                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                stp_df_filtered[group_by_col_name] = pd.Categorical(stp_df_filtered[group_by_col_name], categories=day_order, ordered=True)

                            elif analysis_type_stp == 'Hour of Day':
                                group_by_col_name = 'HourOfDay'
                                stp_df_filtered[group_by_col_name] = stp_df_filtered[date_col_stp].dt.hour
                            
                            st.subheader(f"Sales by {analysis_type_stp}: {aggregation_stp} of '{metric_col_stp if aggregation_stp != 'Count of Orders' else order_id_col_stp_count}'")

                            if aggregation_stp == 'Sum':
                                grouped_data_stp = stp_df_filtered.groupby(group_by_col_name)[metric_col_stp].sum()
                            elif aggregation_stp == 'Mean':
                                grouped_data_stp = stp_df_filtered.groupby(group_by_col_name)[metric_col_stp].mean()
                            elif aggregation_stp == 'Count of Orders':
                                grouped_data_stp = stp_df_filtered.groupby(group_by_col_name)[order_id_col_stp_count].nunique()
                            else:
                                st.error("Invalid aggregation function selected."); st.stop()

                            if analysis_type_stp == 'Day of Week':
                                grouped_data_stp = grouped_data_stp.sort_index()

                            if grouped_data_stp.empty:
                                st.info(f"No data to display for {analysis_type_stp} breakdown.")
                            else:
                                st.markdown(f"#### Sales by {analysis_type_stp}")
                                st.bar_chart(grouped_data_stp)
                                st.dataframe(grouped_data_stp.reset_index())
                    except Exception as e:
                        st.error(f"An error occurred during Time Pattern Analysis: {e}")

        # --- New Tool 1: Top N Movers (Products with Largest Sales Change) ---
        with st.expander("🚀 Top N Movers (Products with Largest Sales Change)", expanded=False):
            st.info("Identify products (SKUs) with the largest sales increase or decrease between two periods. Select SKU, Date, and a sales value (Amount or Qty).")

            all_cols_tnm = df.columns.tolist()
            numeric_cols_tnm = get_numeric_columns(df)
            date_cols_tnm = date_cols

            st.markdown("#### Column Selection")
            col1_tnm, col2_tnm, col3_tnm = st.columns(3)
            with col1_tnm:
                sku_col_tnm = st.selectbox("Select Product ID/SKU column:", all_cols_tnm, index=all_cols_tnm.index('SKU') if 'SKU' in all_cols_tnm else 0, key="tnm_sku")
            with col2_tnm:
                date_col_tnm = st.selectbox("Select Date column:", date_cols_tnm, index=date_cols_tnm.index('Date') if 'Date' in date_cols_tnm else 0, key="tnm_date")
            with col3_tnm:
                value_col_tnm = st.selectbox("Select Sales Value (Amount or Qty):", numeric_cols_tnm, index=numeric_cols_tnm.index('Amount') if 'Amount' in numeric_cols_tnm else (numeric_cols_tnm.index('Qty') if 'Qty' in numeric_cols_tnm else 0), key="tnm_value")

            st.markdown("#### Define Periods for Comparison")
            # Period 1
            st.markdown("**Period 1 (e.g., Previous Period)**")
            p1_col1, p1_col2 = st.columns(2)
            min_date_val_tnm = df[date_col_tnm].min() if date_col_tnm and date_col_tnm in df.columns and not df[date_col_tnm].isnull().all() else datetime.now().date() - pd.Timedelta(days=365*2)
            max_date_val_tnm = df[date_col_tnm].max() if date_col_tnm and date_col_tnm in df.columns and not df[date_col_tnm].isnull().all() else datetime.now().date()
            if isinstance(min_date_val_tnm, pd.Timestamp): min_date_val_tnm = min_date_val_tnm.date()
            if isinstance(max_date_val_tnm, pd.Timestamp): max_date_val_tnm = max_date_val_tnm.date()

            with p1_col1:
                start_date_p1_tnm = st.date_input("Start Date P1:", min_date_val_tnm, min_value=min_date_val_tnm, max_value=max_date_val_tnm, key="tnm_p1_start")
            with p1_col2:
                end_date_p1_tnm = st.date_input("End Date P1:", pd.to_datetime(start_date_p1_tnm) + pd.Timedelta(days=29) if start_date_p1_tnm else min_date_val_tnm, min_value=min_date_val_tnm, max_value=max_date_val_tnm, key="tnm_p1_end")

            # Period 2
            st.markdown("**Period 2 (e.g., Current Period)**")
            p2_col1, p2_col2 = st.columns(2)
            with p2_col1:
                start_date_p2_tnm = st.date_input("Start Date P2:", pd.to_datetime(end_date_p1_tnm) + pd.Timedelta(days=1) if end_date_p1_tnm else min_date_val_tnm, min_value=min_date_val_tnm, max_value=max_date_val_tnm, key="tnm_p2_start")
            with p2_col2:
                end_date_p2_tnm = st.date_input("End Date P2:", pd.to_datetime(start_date_p2_tnm) + pd.Timedelta(days=29) if start_date_p2_tnm else max_date_val_tnm, min_value=min_date_val_tnm, max_value=max_date_val_tnm, key="tnm_p2_end")

            top_n_tnm = st.slider("Number of Top/Bottom Movers to Display:", 5, 25, 10, key="tnm_top_n")

            if st.button("📊 Analyze Top Movers", key="tnm_run"):
                if not all([sku_col_tnm, date_col_tnm, value_col_tnm, start_date_p1_tnm, end_date_p1_tnm, start_date_p2_tnm, end_date_p2_tnm]):
                    st.warning("Please select all columns and define both periods.")
                elif pd.to_datetime(start_date_p1_tnm) > pd.to_datetime(end_date_p1_tnm) or pd.to_datetime(start_date_p2_tnm) > pd.to_datetime(end_date_p2_tnm):
                    st.warning("Start date cannot be after end date for a period.")
                elif pd.to_datetime(end_date_p1_tnm) >= pd.to_datetime(start_date_p2_tnm):
                    st.warning("Period 1 must end before Period 2 begins for a meaningful comparison.")
                else:
                    try:
                        tnm_df_full = df[[sku_col_tnm, date_col_tnm, value_col_tnm]].copy()
                        tnm_df_full[date_col_tnm] = pd.to_datetime(tnm_df_full[date_col_tnm], errors='coerce')
                        tnm_df_full = tnm_df_full.dropna()

                        # Filter data for Period 1 and Period 2
                        p1_data = tnm_df_full[(tnm_df_full[date_col_tnm] >= pd.to_datetime(start_date_p1_tnm)) & (tnm_df_full[date_col_tnm] <= pd.to_datetime(end_date_p1_tnm))]
                        p2_data = tnm_df_full[(tnm_df_full[date_col_tnm] >= pd.to_datetime(start_date_p2_tnm)) & (tnm_df_full[date_col_tnm] <= pd.to_datetime(end_date_p2_tnm))]

                        if p1_data.empty or p2_data.empty:
                            st.warning("No data available for one or both selected periods.")
                        else:
                            sales_p1 = p1_data.groupby(sku_col_tnm)[value_col_tnm].sum().reset_index(name='Sales_P1')
                            sales_p2 = p2_data.groupby(sku_col_tnm)[value_col_tnm].sum().reset_index(name='Sales_P2')

                            merged_sales = pd.merge(sales_p1, sales_p2, on=sku_col_tnm, how='outer').fillna(0)
                            merged_sales['SalesChange'] = merged_sales['Sales_P2'] - merged_sales['Sales_P1']
                            merged_sales['SalesChangePercentage'] = ((merged_sales['Sales_P2'] - merged_sales['Sales_P1']) / merged_sales['Sales_P1'].replace(0, np.nan)) * 100
                            merged_sales['SalesChangePercentage'] = merged_sales['SalesChangePercentage'].fillna(0).replace([np.inf, -np.inf], 0)

                            st.subheader("Top N Product Movers Results")
                            st.markdown(f"#### Top {top_n_tnm} Products with Largest Sales Increase ({value_col_tnm})")
                            top_gainers = merged_sales.sort_values(by='SalesChange', ascending=False).head(top_n_tnm)
                            st.dataframe(top_gainers)

                            st.markdown(f"#### Top {top_n_tnm} Products with Largest Sales Decrease ({value_col_tnm})")
                            top_losers = merged_sales.sort_values(by='SalesChange', ascending=True).head(top_n_tnm)
                            st.dataframe(top_losers)

                    except Exception as e:
                        st.error(f"An error occurred during Top N Movers analysis: {e}")

        # --- New Tool 2: Order Size Distribution Analysis ---
        with st.expander("📦 Order Size Distribution Analysis", expanded=False):
            st.info("Analyze the distribution of order sizes, either by number of unique items per order or total quantity per order.")

            all_cols_osd = df.columns.tolist()
            numeric_cols_osd = get_numeric_columns(df)

            order_id_col_osd = st.selectbox("Select Order ID column:", all_cols_osd, index=all_cols_osd.index('Order ID') if 'Order ID' in all_cols_osd else 0, key="osd_order_id")
            analysis_type_osd = st.radio("Analyze order size by:", ("Number of Unique Items (SKUs)", "Total Quantity of Items"), key="osd_analysis_type")

            item_id_col_osd, qty_col_osd = None, None
            if analysis_type_osd == "Number of Unique Items (SKUs)":
                item_id_col_osd = st.selectbox("Select Product ID/SKU column:", all_cols_osd, index=all_cols_osd.index('SKU') if 'SKU' in all_cols_osd else 0, key="osd_item_id")
            else: # Total Quantity of Items
                qty_col_osd = st.selectbox("Select Quantity Sold column:", numeric_cols_osd, index=numeric_cols_osd.index('Qty') if 'Qty' in numeric_cols_osd else 0, key="osd_qty")

            if st.button("📊 Analyze Order Size Distribution", key="osd_run"):
                if not order_id_col_osd or (analysis_type_osd == "Number of Unique Items (SKUs)" and not item_id_col_osd) or \
                   (analysis_type_osd == "Total Quantity of Items" and not qty_col_osd):
                    st.warning("Please select all required columns for the chosen analysis type.")
                else:
                    try:
                        osd_df = df.copy()
                        if analysis_type_osd == "Number of Unique Items (SKUs)":
                            osd_df = osd_df[[order_id_col_osd, item_id_col_osd]].dropna()
                            order_sizes = osd_df.groupby(order_id_col_osd)[item_id_col_osd].nunique()
                            plot_label = "Number of Unique Items per Order"
                        else: # Total Quantity of Items
                            osd_df = osd_df[[order_id_col_osd, qty_col_osd]].dropna()
                            order_sizes = osd_df.groupby(order_id_col_osd)[qty_col_osd].sum()
                            plot_label = "Total Quantity of Items per Order"

                        if order_sizes.empty:
                            st.warning("No data available to analyze order size distribution.")
                        else:
                            st.subheader(f"Distribution of {plot_label}")
                            fig_osd, ax_osd = plt.subplots()
                            sns.histplot(order_sizes, kde=False, ax=ax_osd, bins=max(10, min(50, order_sizes.nunique()))) # Dynamic bins
                            ax_osd.set_title(f"Distribution of {plot_label}")
                            ax_osd.set_xlabel(plot_label)
                            ax_osd.set_ylabel("Number of Orders")
                            st.pyplot(fig_osd)
                            st.write("Summary Statistics for Order Sizes:")
                            st.dataframe(order_sizes.describe())

                    except Exception as e:
                        st.error(f"An error occurred during Order Size Distribution analysis: {e}")

        # --- New Tool 3: Sales Lag Analysis (Time Between Purchases) ---
        with st.expander("⏳ Sales Lag Analysis (Time Between Purchases)", expanded=False):
            st.info("For entities making multiple purchases (using 'Order ID' as a proxy if Customer ID isn't available), analyze the time (days) between consecutive transactions.")

            all_cols_sla = df.columns.tolist()
            date_cols_sla = date_cols

            # Using 'Order ID' as a proxy for "customer" if no dedicated Customer ID exists.
            # This means we are analyzing time between transactions for the same 'Order ID' if it can appear multiple times on different dates.
            # However, 'Order ID' is unique per row in this dataset. So, this tool is more conceptual for this specific dataset
            # unless a true Customer ID is present or 'Order ID' could represent a recurring subscription ID.
            # For demonstration, we'll proceed assuming 'Order ID' could represent an entity making multiple transactions over time.
            entity_id_col_sla = st.selectbox("Select Customer/Entity ID column (e.g., a true Customer ID or 'Order ID' as proxy):", all_cols_sla, index=all_cols_sla.index('Order ID') if 'Order ID' in all_cols_sla else 0, key="sla_entity_id")
            date_col_sla = st.selectbox("Select Order Date column:", date_cols_sla, index=date_cols_sla.index('Date') if 'Date' in date_cols_sla else 0, key="sla_date")

            if st.button("📊 Analyze Sales Lag", key="sla_run"):
                if not entity_id_col_sla or not date_col_sla:
                    st.warning("Please select both Entity ID and Date columns.")
                else:
                    try:
                        sla_df = df[[entity_id_col_sla, date_col_sla]].copy()
                        sla_df[date_col_sla] = pd.to_datetime(sla_df[date_col_sla], errors='coerce')
                        sla_df = sla_df.dropna().sort_values(by=[entity_id_col_sla, date_col_sla])

                        if sla_df.empty:
                            st.warning("No data for sales lag analysis.")
                        else:
                            sla_df['PreviousPurchaseDate'] = sla_df.groupby(entity_id_col_sla)[date_col_sla].shift(1)
                            sla_df['DaysBetweenPurchases'] = (sla_df[date_col_sla] - sla_df['PreviousPurchaseDate']).dt.days
                            
                            purchase_lags = sla_df.dropna(subset=['DaysBetweenPurchases'])

                            if purchase_lags.empty:
                                st.info(f"No repeat purchases found for entities in '{entity_id_col_sla}' to calculate lag times. This tool is most effective with a true Customer ID and repeat purchases.")
                            else:
                                st.subheader("Sales Lag Analysis Results")
                                st.markdown("###### Distribution of Days Between Consecutive Purchases")
                                fig_sla, ax_sla = plt.subplots()
                                sns.histplot(purchase_lags['DaysBetweenPurchases'], kde=True, ax=ax_sla, bins=30)
                                ax_sla.set_title("Distribution of Days Between Purchases")
                                ax_sla.set_xlabel("Days Between Purchases")
                                st.pyplot(fig_sla)

                                st.write("Summary Statistics for Days Between Purchases:")
                                st.dataframe(purchase_lags['DaysBetweenPurchases'].describe())
                                st.caption(f"Note: If '{entity_id_col_sla}' is 'Order ID' (unique per row), this analysis might not yield meaningful results unless an 'Order ID' can truly represent an entity with multiple dated transactions.")
                    except Exception as e:
                        st.error(f"An error occurred during Sales Lag Analysis: {e}")

        # --- New Tool 4: Category Penetration by Sales Channel ---
        with st.expander("🔗 Category Penetration by Sales Channel", expanded=False):
            st.info("Analyze the revenue share of each Product Category within different Sales Channels. Helps understand channel-specific category strengths.")

            all_cols_cpsc = df.columns.tolist()
            numeric_cols_cpsc = get_numeric_columns(df)
            categorical_cols_cpsc = get_categorical_columns(df)

            sales_channel_col_cpsc = st.selectbox("Select Sales Channel column:", categorical_cols_cpsc, index=categorical_cols_cpsc.index('Sales Channel') if 'Sales Channel' in categorical_cols_cpsc else 0, key="cpsc_channel")
            category_col_cpsc = st.selectbox("Select Category column:", categorical_cols_cpsc, index=categorical_cols_cpsc.index('Category') if 'Category' in categorical_cols_cpsc else 0, key="cpsc_category")
            amount_col_cpsc = st.selectbox("Select Sales Amount column:", numeric_cols_cpsc, index=numeric_cols_cpsc.index('Amount') if 'Amount' in numeric_cols_cpsc else 0, key="cpsc_amount")

            if st.button("📊 Analyze Category Penetration", key="cpsc_run"):
                if not all([sales_channel_col_cpsc, category_col_cpsc, amount_col_cpsc]):
                    st.warning("Please select Sales Channel, Category, and Amount columns.")
                else:
                    try:
                        cpsc_df = df[[sales_channel_col_cpsc, category_col_cpsc, amount_col_cpsc]].copy().dropna()
                        if cpsc_df.empty:
                            st.warning("No data for category penetration analysis.")
                        else:
                            pivot_cpsc = pd.pivot_table(cpsc_df, values=amount_col_cpsc, index=sales_channel_col_cpsc, columns=category_col_cpsc, aggfunc='sum', fill_value=0)
                            penetration_cpsc = pivot_cpsc.apply(lambda x: (x / x.sum() * 100), axis=1) # Percentage along rows (channels)

                            st.subheader(f"Category Revenue Share (%) within each {sales_channel_col_cpsc}")
                            st.dataframe(penetration_cpsc.style.format("{:.1f}%"))

                            if not penetration_cpsc.empty:
                                # Select top N categories overall for a cleaner stacked bar chart
                                top_n_cat_plot = 7
                                top_categories_overall = cpsc_df.groupby(category_col_cpsc)[amount_col_cpsc].sum().nlargest(top_n_cat_plot).index
                                plot_df_cpsc = penetration_cpsc[top_categories_overall if not top_categories_overall.empty else penetration_cpsc.columns[:top_n_cat_plot]]
                                
                                if not plot_df_cpsc.empty:
                                    fig_cpsc, ax_cpsc = plt.subplots(figsize=(12, 7))
                                    plot_df_cpsc.plot(kind='bar', stacked=True, ax=ax_cpsc, colormap='viridis')
                                    ax_cpsc.set_title(f'Category Revenue Share by {sales_channel_col_cpsc} (Top Categories)')
                                    ax_cpsc.set_ylabel('Percentage of Channel Revenue (%)')
                                    ax_cpsc.legend(title=category_col_cpsc, bbox_to_anchor=(1.05, 1), loc='upper left')
                                    plt.xticks(rotation=45, ha="right")
                                    plt.tight_layout()
                                    st.pyplot(fig_cpsc)
                                else:
                                    st.info("Not enough data to plot category penetration.")
                    except Exception as e:
                        st.error(f"An error occurred during Category Penetration analysis: {e}")

        # --- New Tool 5: Shipment Status Tracker ---
        with st.expander("🚚 Shipment Status Tracker", expanded=False):
            st.info("Track order shipment statuses (e.g., 'Shipped', 'Cancelled', 'Delivered') over time or by service level. Uses 'Status' or 'Courier Status' column.")

            all_cols_sst = df.columns.tolist()
            date_cols_sst = date_cols
            categorical_cols_sst = get_categorical_columns(df)

            order_id_col_sst = st.selectbox("Select Order ID column:", all_cols_sst, index=all_cols_sst.index('Order ID') if 'Order ID' in all_cols_sst else 0, key="sst_order_id")
            # Allow choosing between 'Status' and 'Courier Status'
            status_col_options_sst = [col for col in ['Status', 'Courier Status'] if col in all_cols_sst]
            if not status_col_options_sst: status_col_options_sst = all_cols_sst # Fallback if specific ones not found
            
            status_col_sst = st.selectbox("Select Status column ('Status' or 'Courier Status'):", status_col_options_sst, key="sst_status")
            date_col_sst = st.selectbox("Select Date column (for trends):", date_cols_sst, index=date_cols_sst.index('Date') if 'Date' in date_cols_sst else 0, key="sst_date")
            grouping_col_sst = st.selectbox("Optional: Group by column (e.g., ship-service-level):", [None] + categorical_cols_sst, key="sst_grouping")
            agg_freq_sst = st.selectbox("Aggregate trend by:", ["D", "W", "M"], index=1, format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x], key="sst_agg_freq")

            if st.button("📊 Track Shipment Statuses", key="sst_run"):
                if not order_id_col_sst or not status_col_sst or not date_col_sst:
                    st.warning("Please select Order ID, Status, and Date columns.")
                elif status_col_sst not in df.columns: # Check if selected status column actually exists
                    st.error(f"The selected status column '{status_col_sst}' was not found in the dataset.")
                else:
                    try:
                        sst_df = df[[order_id_col_sst, status_col_sst, date_col_sst]].copy()
                        if grouping_col_sst and grouping_col_sst in df.columns:
                            sst_df[grouping_col_sst] = df[grouping_col_sst]
                        
                        sst_df[date_col_sst] = pd.to_datetime(sst_df[date_col_sst], errors='coerce')
                        sst_df = sst_df.dropna(subset=[order_id_col_sst, status_col_sst, date_col_sst])

                        if sst_df.empty:
                            st.warning("No data for shipment status tracking.")
                        else:
                            st.subheader(f"Shipment Status Analysis (based on '{status_col_sst}')")

                            st.markdown("###### Current Distribution of Shipment Statuses")
                            status_counts = sst_df[status_col_sst].value_counts()
                            st.bar_chart(status_counts)
                            st.dataframe(status_counts.reset_index())

                            st.markdown(f"###### Trend of Shipment Statuses Over Time ({agg_freq_sst})")
                            sst_df['TimePeriod'] = sst_df[date_col_sst].dt.to_period(agg_freq_sst)
                            
                            if grouping_col_sst and grouping_col_sst in sst_df.columns:
                                trend_data = sst_df.groupby(['TimePeriod', grouping_col_sst, status_col_sst])[order_id_col_sst].nunique().unstack(fill_value=0).unstack(fill_value=0)
                                st.write(f"Showing trends for top statuses, grouped by '{grouping_col_sst}'. Select a group to view its trend:")
                                available_groups_sst = sst_df[grouping_col_sst].dropna().unique().tolist()
                                selected_group_trend_sst = st.selectbox("Select group to display trend:", [None] + available_groups_sst, key="sst_select_group_trend")
                                if selected_group_trend_sst and selected_group_trend_sst in trend_data.columns.levels[0]:
                                    st.line_chart(trend_data[selected_group_trend_sst].fillna(0))
                                elif selected_group_trend_sst is None and not trend_data.empty:
                                    st.info(f"Select a group from the dropdown to see its specific status trend. Displaying overall status counts above.")
                                elif not trend_data.empty : # if selected group not in columns (should not happen with proper list)
                                    st.info(f"No trend data for group '{selected_group_trend_sst}'.")
                            else:
                                trend_data = sst_df.groupby(['TimePeriod', status_col_sst])[order_id_col_sst].nunique().unstack(fill_value=0)
                                if not trend_data.empty:
                                    st.line_chart(trend_data.fillna(0))
                                else:
                                    st.info("No trend data to display.")
                            
                            if not trend_data.empty:
                                st.write("Trend Data (first 10 periods):")
                                st.dataframe(trend_data.head(10))

                    except Exception as e:
                        st.error(f"An error occurred during Shipment Status Tracking: {e}")

        # --- New Tool 6 (Total 18): Period-over-Period (PoP) Growth Analysis ---
        with st.expander("📈 Period-over-Period (PoP) Growth Analysis", expanded=False):
            st.info("Calculate and visualize key growth metrics (e.g., Month-over-Month, Year-over-Year) for Sales Revenue, Number of Orders, and Average Order Value (AOV).")

            all_cols_pop = df.columns.tolist()
            numeric_cols_pop = get_numeric_columns(df)
            date_cols_pop = date_cols

            st.markdown("#### Column Selection")
            pop_col1, pop_col2, pop_col3 = st.columns(3)
            with pop_col1:
                date_col_pop = st.selectbox("Select Date column:", date_cols_pop, index=date_cols_pop.index('Date') if 'Date' in date_cols_pop else 0, key="pop_date")
            with pop_col2:
                amount_col_pop = st.selectbox("Select Sales Amount column:", numeric_cols_pop, index=numeric_cols_pop.index('Amount') if 'Amount' in numeric_cols_pop else 0, key="pop_amount")
            with pop_col3:
                order_id_col_pop = st.selectbox("Select Order ID column (for order counts & AOV):", all_cols_pop, index=all_cols_pop.index('Order ID') if 'Order ID' in all_cols_pop else 0, key="pop_order_id")

            pop_period_type = st.selectbox("Select PoP Type:", ["Month-over-Month (MoM)", "Quarter-over-Quarter (QoQ)", "Year-over-Year (YoY)"], key="pop_period_type")

            if st.button("📊 Analyze PoP Growth", key="pop_run"):
                if not all([date_col_pop, amount_col_pop, order_id_col_pop]):
                    st.warning("Please select Date, Amount, and Order ID columns.")
                else:
                    try:
                        pop_df = df[[date_col_pop, amount_col_pop, order_id_col_pop]].copy()
                        pop_df[date_col_pop] = pd.to_datetime(pop_df[date_col_pop], errors='coerce')
                        pop_df = pop_df.dropna()

                        if pop_df.empty:
                            st.warning("No data available for PoP growth analysis.")
                        else:
                            # Aggregate data by the chosen period start
                            if pop_period_type == "Month-over-Month (MoM)":
                                freq_code = 'MS' # Month Start
                                shift_periods = 1
                            elif pop_period_type == "Quarter-over-Quarter (QoQ)":
                                freq_code = 'QS' # Quarter Start
                                shift_periods = 1
                            else: # Year-over-Year (YoY)
                                freq_code = 'YS' # Year Start
                                shift_periods = 1
                            
                            # Ensure index is datetime for resampling
                            pop_df_indexed = pop_df.set_index(date_col_pop)

                            # Aggregate metrics
                            periodic_sales = pop_df_indexed.resample(freq_code)[amount_col_pop].sum()
                            periodic_orders = pop_df_indexed.resample(freq_code)[order_id_col_pop].nunique()
                            periodic_aov = periodic_sales / periodic_orders.replace(0, np.nan)

                            if periodic_sales.empty or len(periodic_sales) < shift_periods + 1:
                                st.warning(f"Not enough historical periods to calculate {pop_period_type} growth. Need at least {shift_periods+1} periods.")
                            else:
                                growth_data = pd.DataFrame({
                                    'TotalSales': periodic_sales,
                                    'TotalOrders': periodic_orders,
                                    'AOV': periodic_aov
                                }).dropna()

                                # Calculate PoP growth
                                for col in ['TotalSales', 'TotalOrders', 'AOV']:
                                    growth_data[f'{col}_Previous'] = growth_data[col].shift(shift_periods)
                                    growth_data[f'{col}_Growth_%'] = ((growth_data[col] - growth_data[f'{col}_Previous']) / growth_data[f'{col}_Previous'].replace(0, np.nan)) * 100
                                
                                growth_data = growth_data.dropna(subset=[f'{col}_Growth_%' for col in ['TotalSales', 'TotalOrders', 'AOV']], how='all')
                                growth_data = growth_data.replace([np.inf, -np.inf], np.nan) # Handle infinities from division by zero

                                st.subheader(f"{pop_period_type} Growth Analysis Results")
                                st.dataframe(growth_data)

                                st.markdown(f"#### Sales Growth ({pop_period_type})")
                                st.line_chart(growth_data[[f'TotalSales_Growth_%']].fillna(0))
                                
                                st.markdown(f"#### Order Count Growth ({pop_period_type})")
                                st.line_chart(growth_data[[f'TotalOrders_Growth_%']].fillna(0))

                                st.markdown(f"#### AOV Growth ({pop_period_type})")
                                st.line_chart(growth_data[[f'AOV_Growth_%']].fillna(0))
                    except Exception as e:
                        st.error(f"An error occurred during PoP Growth Analysis: {e}")

        # --- New Tool 7 (Total 19): Sales Performance by Price Tiers ---
        with st.expander("💲 Sales Performance by Price Tiers", expanded=False):
            st.info("Define price buckets and analyze sales performance (Revenue, Quantity, Unique Products) within these tiers.")

            all_cols_sppt = df.columns.tolist()
            numeric_cols_sppt = get_numeric_columns(df)

            st.markdown("#### Column Selection")
            sppt_col1, sppt_col2, sppt_col3 = st.columns(3)
            with sppt_col1:
                # Price column could be 'Amount' if each row is an item with its price, or a dedicated 'Price' column
                price_col_sppt = st.selectbox("Select Price column (e.g., Amount per item):", numeric_cols_sppt, index=numeric_cols_sppt.index('Amount') if 'Amount' in numeric_cols_sppt else 0, key="sppt_price")
            with sppt_col2:
                qty_col_sppt = st.selectbox("Select Quantity Sold column:", numeric_cols_sppt, index=numeric_cols_sppt.index('Qty') if 'Qty' in numeric_cols_sppt else 0, key="sppt_qty")
            with sppt_col3:
                sku_col_sppt = st.selectbox("Select Product ID/SKU column (for unique product count):", all_cols_sppt, index=all_cols_sppt.index('SKU') if 'SKU' in all_cols_sppt else 0, key="sppt_sku")

            st.markdown("#### Define Price Tiers")
            st.caption("Enter comma-separated upper bounds for your price tiers (e.g., 20,50,100,500). This will create tiers like <20, 20-50, 50-100, 100-500, >500.")
            tier_bounds_str = st.text_input("Price Tier Upper Bounds:", "20,50,100,200,500", key="sppt_tiers")

            if st.button("📊 Analyze by Price Tiers", key="sppt_run"):
                if not all([price_col_sppt, qty_col_sppt, sku_col_sppt, tier_bounds_str]):
                    st.warning("Please select all columns and define price tiers.")
                else:
                    try:
                        tier_bounds = sorted([float(b.strip()) for b in tier_bounds_str.split(',') if b.strip()])
                        if not tier_bounds:
                            st.warning("Please enter valid numeric price tier bounds.")
                        else:
                            sppt_df = df[[price_col_sppt, qty_col_sppt, sku_col_sppt]].copy().dropna()
                            sppt_df['Revenue'] = sppt_df[price_col_sppt] * sppt_df[qty_col_sppt] # Assuming price_col is per unit price

                            # Create price tier labels
                            bins = [-np.inf] + tier_bounds + [np.inf]
                            labels = [f"< {tier_bounds[0]}"] + \
                                     [f"{tier_bounds[i]}-{tier_bounds[i+1]}" for i in range(len(tier_bounds)-1)] + \
                                     [f"> {tier_bounds[-1]}"]
                            
                            sppt_df['PriceTier'] = pd.cut(sppt_df[price_col_sppt], bins=bins, labels=labels, right=False)

                            if sppt_df.empty:
                                st.warning("No data available for price tier analysis.")
                            else:
                                tier_summary = sppt_df.groupby('PriceTier', observed=False).agg(
                                    TotalRevenue=('Revenue', 'sum'),
                                    TotalQuantity=(qty_col_sppt, 'sum'),
                                    UniqueProducts=(sku_col_sppt, 'nunique')
                                ).reset_index()

                                st.subheader("Sales Performance by Price Tiers")
                                st.dataframe(tier_summary)

                                fig_sppt_rev, ax_sppt_rev = plt.subplots()
                                sns.barplot(data=tier_summary, x='PriceTier', y='TotalRevenue', ax=ax_sppt_rev, palette="viridis")
                                ax_sppt_rev.set_title("Total Revenue by Price Tier")
                                ax_sppt_rev.set_ylabel("Total Revenue")
                                plt.xticks(rotation=45, ha="right")
                                plt.tight_layout()
                                st.pyplot(fig_sppt_rev)

                                fig_sppt_qty, ax_sppt_qty = plt.subplots()
                                sns.barplot(data=tier_summary, x='PriceTier', y='TotalQuantity', ax=ax_sppt_qty, palette="magma")
                                ax_sppt_qty.set_title("Total Quantity Sold by Price Tier")
                                ax_sppt_qty.set_ylabel("Total Quantity")
                                plt.xticks(rotation=45, ha="right")
                                plt.tight_layout()
                                st.pyplot(fig_sppt_qty)

                    except ValueError:
                        st.error("Invalid input for price tier bounds. Please use comma-separated numbers.")
                    except Exception as e:
                        st.error(f"An error occurred during Price Tier Analysis: {e}")

        # --- New Tool 8 (Total 20): Comparative Profile of Top vs. Bottom Performing Products ---
        with st.expander("🆚 Comparative Profile of Top vs. Bottom Products", expanded=False):
            st.info("Compare characteristics of top N% vs. bottom N% of products (by revenue) across various attributes like Sales Channel, Fulfilment, B2B status, etc.")

            all_cols_cptb = df.columns.tolist()
            numeric_cols_cptb = get_numeric_columns(df)
            categorical_cols_cptb = get_categorical_columns(df, nunique_threshold=50) # Allow slightly higher cardinality for profiling

            st.markdown("#### Column Selection")
            cptb_col1, cptb_col2 = st.columns(2)
            with cptb_col1:
                sku_col_cptb = st.selectbox("Select Product ID/SKU column:", all_cols_cptb, index=all_cols_cptb.index('SKU') if 'SKU' in all_cols_cptb else 0, key="cptb_sku")
                amount_col_cptb = st.selectbox("Select Sales Amount column (for ranking):", numeric_cols_cptb, index=numeric_cols_cptb.index('Amount') if 'Amount' in numeric_cols_cptb else 0, key="cptb_amount")
            with cptb_col2:
                profiling_attributes_cptb = st.multiselect(
                    "Select attributes for profiling:",
                    [col for col in categorical_cols_cptb + numeric_cols_cptb if col not in [sku_col_cptb, amount_col_cptb]],
                    default=[col for col in ['Sales Channel', 'Fulfilment', 'B2B', 'Category', 'Qty'] if col in df.columns and col not in [sku_col_cptb, amount_col_cptb]],
                    key="cptb_attributes"
                )

            percentile_cptb = st.slider("Select N% for Top/Bottom (e.g., 20% for top 20% and bottom 20%):", 5, 50, 20, 5, key="cptb_percentile")

            if st.button("📊 Compare Product Profiles", key="cptb_run"):
                if not all([sku_col_cptb, amount_col_cptb]) or not profiling_attributes_cptb:
                    st.warning("Please select SKU, Amount, and at least one Profiling Attribute.")
                else:
                    try:
                        cptb_df = df.copy()
                        product_revenue = cptb_df.groupby(sku_col_cptb)[amount_col_cptb].sum().reset_index()
                        
                        if product_revenue.empty:
                            st.warning("No product revenue data to rank products.")
                        else:
                            num_products = len(product_revenue)
                            top_n_count = int(num_products * (percentile_cptb / 100.0))
                            
                            if top_n_count == 0 and num_products > 0: # Ensure at least 1 product if percentile is too small
                                top_n_count = 1
                            
                            if num_products < 2 or top_n_count == 0:
                                st.warning(f"Not enough products ({num_products}) or too small a percentile to form distinct top/bottom groups.")
                            else:
                                top_products = product_revenue.nlargest(top_n_count, amount_col_cptb)[sku_col_cptb]
                                bottom_products = product_revenue.nsmallest(top_n_count, amount_col_cptb)[sku_col_cptb]

                                top_df = cptb_df[cptb_df[sku_col_cptb].isin(top_products)]
                                bottom_df = cptb_df[cptb_df[sku_col_cptb].isin(bottom_products)]

                                if top_df.empty or bottom_df.empty:
                                    st.warning("Could not form both top and bottom product groups. Check data or percentile.")
                                else:
                                    st.subheader(f"Comparative Profile: Top {percentile_cptb}% vs. Bottom {percentile_cptb}% Products")

                                    profile_summary_list = []
                                    for attr in profiling_attributes_cptb:
                                        if attr in cptb_df.columns:
                                            if cptb_df[attr].dtype == 'object' or cptb_df[attr].nunique() < 20 : # Categorical or low-cardinality numeric
                                                top_dist = top_df[attr].value_counts(normalize=True).mul(100).round(1)
                                                bottom_dist = bottom_df[attr].value_counts(normalize=True).mul(100).round(1)
                                                
                                                # Combine into a displayable format
                                                comparison_df = pd.DataFrame({
                                                    f'Top {percentile_cptb}% Dist. (%)': top_dist,
                                                    f'Bottom {percentile_cptb}% Dist. (%)': bottom_dist
                                                }).fillna(0)
                                                
                                                st.markdown(f"###### Attribute: {attr} (Distribution %)")
                                                st.dataframe(comparison_df)
                                                
                                                # Plot if not too many categories
                                                if len(comparison_df) < 10 and len(comparison_df)>0:
                                                    fig_comp, ax_comp = plt.subplots()
                                                    comparison_df.plot(kind='bar', ax=ax_comp, figsize=(8, max(4, len(comparison_df)*0.5)))
                                                    ax_comp.set_title(f"Distribution of {attr}")
                                                    ax_comp.set_ylabel("Percentage (%)")
                                                    plt.xticks(rotation=45, ha="right")
                                                    plt.tight_layout()
                                                    st.pyplot(fig_comp)

                                            elif pd.api.types.is_numeric_dtype(cptb_df[attr]): # Numeric attribute
                                                top_mean = top_df[attr].mean()
                                                bottom_mean = bottom_df[attr].mean()
                                                top_median = top_df[attr].median()
                                                bottom_median = bottom_df[attr].median()
                                                profile_summary_list.append({
                                                    'Attribute': attr,
                                                    f'Top {percentile_cptb}% Mean': f"{top_mean:.2f}",
                                                    f'Bottom {percentile_cptb}% Mean': f"{bottom_mean:.2f}",
                                                    f'Top {percentile_cptb}% Median': f"{top_median:.2f}",
                                                    f'Bottom {percentile_cptb}% Median': f"{bottom_median:.2f}",
                                                })
                                    
                                    if profile_summary_list:
                                        st.markdown("###### Numeric Attribute Comparison (Mean/Median)")
                                        st.dataframe(pd.DataFrame(profile_summary_list).set_index('Attribute'))

                    except Exception as e:
                        st.error(f"An error occurred during Comparative Profile Analysis: {e}")

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
                                    
                                    log_reg_model = sm.Logit(y_train, X_train.astype(float)).fit(disp=0) # disp=0 to suppress convergence messages
                                    st.subheader("Logistic Regression Results")

                                    # Displaying results in a more organized way
                                    st.markdown("#### Model Summary Statistics")
                                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                                    try:
                                        col_metric1.metric("Pseudo R-squared", f"{log_reg_model.prsquared:.4f}")
                                    except AttributeError: # prsquared might not be available for all fits
                                        col_metric1.metric("Pseudo R-squared", "N/A")
                                    col_metric2.metric("Log-Likelihood", f"{log_reg_model.llf:.2f}")
                                    col_metric3.metric("LL-Null", f"{log_reg_model.llnull:.2f}")

                                    model_method_display = "N/A" # Default value
                                    try:
                                        model_method_display = log_reg_model.model.method
                                    except AttributeError:
                                        # This handles the specific error you encountered.
                                        st.warning("Note: Could not automatically retrieve the model fitting method. This is unexpected.")
                                    st.write(f"**Method:** {model_method_display}")
                                    st.write(f"**Number of Observations:** {log_reg_model.nobs}")
                                    # st.write(f"**AIC:** {log_reg_model.aic:.2f}") # Available
                                    # st.write(f"**BIC:** {log_reg_model.bic:.2f}") # Available

                                    st.markdown("#### Coefficients")
                                    # summary2() provides tables that are easier to parse
                                    summary_tables = log_reg_model.summary2().tables
                                    if len(summary_tables) > 1:
                                        st.dataframe(summary_tables[1]) # The second table is usually coefficients
                                    else: # Fallback to plain text if summary2 structure is unexpected
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
                                uplift_rel = pm.Deterministic('uplift_rel', pm.math.switch(pm.math.gt(p_A, 1e-9), (p_B - p_A) / p_A, 0.0))
                                
                                # Use return_inferencedata=True for ArviZ compatibility
                                trace = pm.sample(2000, tune=1000, cores=1, progressbar=True, return_inferencedata=True)

                            # Assuming trace.posterior might be a method in the environment causing the error.
                            # If it's a property (as standard), calling it would error.
                            # If it's a method (as the error implies), it needs to be called.
                            posterior_dataset = trace.posterior
                            try:
                                # Attempt to call if it's a method, to resolve the "method object is not subscriptable" error.
                                # If it's already a dataset (property), this call will fail, and we use it directly.
                                posterior_dataset = trace.posterior()
                            except TypeError: # Likely "'Dataset' object is not callable" if it was already a property
                                pass # Use trace.posterior as is

                            st.subheader("Bayesian A/B Test Results")
                            
                            fig_posterior, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=False) # sharex=False might be better for different scales
                            az.plot_posterior(posterior_dataset, var_names=[f'p_{group_names_bab[0]}'], ax=axes[0], show=False, hdi_prob=0.95)
                            az.plot_posterior(posterior_dataset, var_names=[f'p_{group_names_bab[1]}'], ax=axes[1], show=False, hdi_prob=0.95)
                            az.plot_posterior(posterior_dataset, var_names=['delta'], ax=axes[2], show=False, hdi_prob=0.95, ref_val=0)
                            az.plot_posterior(posterior_dataset, var_names=['uplift_rel'], ax=axes[3], show=False, hdi_prob=0.95, ref_val=0)
                            
                            axes[0].set_title(f'Posterior of rate for {group_names_bab[0]}')
                            axes[1].set_title(f'Posterior of rate for {group_names_bab[1]}')
                            axes[2].set_title(f'Posterior of absolute difference ({group_names_bab[1]} - {group_names_bab[0]})')
                            axes[3].set_title(f'Posterior of relative uplift ({group_names_bab[1]} vs {group_names_bab[0]})')
                            plt.tight_layout()
                            st.pyplot(fig_posterior)
                            prob_b_better_a_bab = (posterior_dataset['delta'].values > 0).mean()
                            st.write(f"Probability that Group '{group_names_bab[1]}'s rate is greater than Group '{group_names_bab[0]}'s: {prob_b_better_a_bab:.2%}")
                            hdi_delta = az.hdi(posterior_dataset['delta'], hdi_prob=0.95).values
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
        
        # Category 2: Machine Learning - Unsupervised (MLU)
        with st.expander("🧠 Machine Learning - Unsupervised (MLU)", expanded=False): # Default to not expanded
            st.info("Unsupervised learning techniques to discover patterns and structures in your data without predefined labels.")

            # MLU 1: Anomaly Detection (Isolation Forest)
            st.subheader("MLU 1: Anomaly Detection (Isolation Forest)")
            st.markdown("Identify unusual data points (anomalies) in your sales data based on selected numeric features. This can help detect outliers like exceptionally high/low sales amounts or quantities.")
            
            numeric_cols_mlu1 = get_numeric_columns(df)
            if len(numeric_cols_mlu1) < 1:
                st.warning("Anomaly detection requires at least one numeric column.")
            else:
                features_mlu1 = st.multiselect("Select numeric features for anomaly detection:", 
                                               numeric_cols_mlu1, 
                                               default=[col for col in ['Amount', 'Qty'] if col in numeric_cols_mlu1], 
                                               key="mlu1_features")
                contamination_mlu1 = st.slider("Expected proportion of outliers (contamination):", 0.01, 0.2, 0.05, 0.01, key="mlu1_contamination",
                                               help="Adjust this based on how many anomalies you expect. Higher means more points flagged.")

                if st.button("🔍 Run Anomaly Detection", key="mlu1_run"):
                    if not features_mlu1:
                        st.warning("Please select at least one feature for anomaly detection.")
                    else:
                        try:
                            mlu1_df = df[features_mlu1].copy().dropna()
                            if mlu1_df.empty or len(mlu1_df) < 2:
                                st.warning("Not enough data after dropping NaNs for selected features.")
                            else:
                                model_iso = IsolationForest(contamination=contamination_mlu1, random_state=42)
                                mlu1_df['anomaly_score'] = model_iso.fit_predict(mlu1_df) # -1 for anomalies, 1 for inliers
                                mlu1_df['is_anomaly'] = mlu1_df['anomaly_score'] == -1
                                
                                anomalies_detected = mlu1_df[mlu1_df['is_anomaly']]
                                
                                st.markdown("##### Anomaly Detection Results") # Changed subheader level
                                st.write(f"Number of anomalies detected: {len(anomalies_detected)} out of {len(mlu1_df)} data points.")
                                st.write("Detected Anomalies (first 100):")
                                st.dataframe(df.loc[anomalies_detected.index].head(100)) # Show original data for anomalies

                                if len(features_mlu1) == 2 and not anomalies_detected.empty:
                                    st.markdown("###### Scatter Plot of Anomalies") # Changed subheader level
                                    fig_mlu1, ax_mlu1 = plt.subplots()
                                    sns.scatterplot(data=mlu1_df, x=features_mlu1[0], y=features_mlu1[1], hue='is_anomaly', palette={True: 'red', False: 'blue'}, ax=ax_mlu1, s=20)
                                    ax_mlu1.set_title(f"Anomaly Detection: {features_mlu1[0]} vs {features_mlu1[1]}")
                                    st.pyplot(fig_mlu1)
                                elif len(features_mlu1) > 2 and not anomalies_detected.empty:
                                    st.info("Plotting first two selected features for anomaly visualization.")
                                    fig_mlu1, ax_mlu1 = plt.subplots()
                                    sns.scatterplot(data=mlu1_df, x=features_mlu1[0], y=features_mlu1[1], hue='is_anomaly', palette={True: 'red', False: 'blue'}, ax=ax_mlu1, s=20)
                                    ax_mlu1.set_title(f"Anomaly Detection (showing {features_mlu1[0]} vs {features_mlu1[1]})")
                                    st.pyplot(fig_mlu1)
                                elif anomalies_detected.empty:
                                    st.info("No anomalies detected with the current settings.")

                        except Exception as e:
                            st.error(f"Error during anomaly detection: {e}")
            st.markdown("---") # Separator between tools

            # MLU 2: Advanced Customer/Order Segmentation (K-Means)
            st.subheader("MLU 2: Advanced Order Segmentation (K-Means)")
            st.markdown("Segment orders based on selected numeric features using K-Means clustering. This can help identify distinct groups of orders with similar characteristics (e.g., high value, low quantity).")
            
            numeric_cols_mlu2 = get_numeric_columns(df)
            # For this dataset, 'Order ID' is unique per row, so segmenting individual transactions/items.
            # If 'Order ID' could group multiple items, we'd aggregate first.

            if len(numeric_cols_mlu2) < 1:
                st.warning("Segmentation requires at least one numeric column.")
            else:
                features_mlu2 = st.multiselect("Select numeric features for segmentation:", 
                                               numeric_cols_mlu2, 
                                               default=[col for col in ['Amount', 'Qty'] if col in numeric_cols_mlu2], 
                                               key="mlu2_features")
                n_clusters_mlu2 = st.slider("Number of segments (clusters):", 2, 10, 3, key="mlu2_n_clusters")

                if st.button("🧩 Run Order Segmentation", key="mlu2_run"):
                    if not features_mlu2:
                        st.warning("Please select at least one feature for segmentation.")
                    else:
                        try:
                            mlu2_df_features = df[features_mlu2].copy().dropna()
                            if mlu2_df_features.empty or len(mlu2_df_features) < n_clusters_mlu2:
                                st.warning(f"Not enough data after dropping NaNs for selected features, or fewer data points ({len(mlu2_df_features)}) than clusters ({n_clusters_mlu2}).")
                            else:
                                scaler_mlu2 = StandardScaler()
                                scaled_features_mlu2 = scaler_mlu2.fit_transform(mlu2_df_features)
                                
                                kmeans_mlu2 = KMeans(n_clusters=n_clusters_mlu2, random_state=42, n_init='auto')
                                # Create a new DataFrame for results to avoid modifying original df directly in this scope
                                result_df_mlu2 = df.loc[mlu2_df_features.index].copy()
                                result_df_mlu2['Segment'] = kmeans_mlu2.fit_predict(scaled_features_mlu2)
                                
                                st.markdown("##### Order Segmentation Results") # Changed subheader level
                                st.write(f"Orders segmented into {n_clusters_mlu2} groups.")
                                st.write("Segment Profiles (Mean values of features):")
                                st.dataframe(result_df_mlu2.groupby('Segment')[features_mlu2].mean())
                                
                                st.write("Segment Sizes:")
                                st.dataframe(result_df_mlu2['Segment'].value_counts().sort_index())

                                if len(features_mlu2) >= 2:
                                    st.markdown("###### Scatter Plot of Segments") # Changed subheader level
                                    fig_mlu2, ax_mlu2 = plt.subplots()
                                    # Use original (unscaled) features for plotting for interpretability, but color by segment
                                    plot_data_mlu2 = result_df_mlu2.copy()
                                    plot_data_mlu2[features_mlu2[0]] = mlu2_df_features[features_mlu2[0]] # ensure original values
                                    plot_data_mlu2[features_mlu2[1]] = mlu2_df_features[features_mlu2[1]]

                                    sns.scatterplot(data=plot_data_mlu2, x=features_mlu2[0], y=features_mlu2[1], hue='Segment', palette='viridis', ax=ax_mlu2, s=20)
                                    ax_mlu2.set_title(f"Order Segments: {features_mlu2[0]} vs {features_mlu2[1]}")
                                    st.pyplot(fig_mlu2)
                                
                                st.write("Sample Data with Segments (first 100):")
                                st.dataframe(result_df_mlu2[[*features_mlu2, 'Segment']].head(100))

                        except Exception as e:
                            st.error(f"Error during segmentation: {e}")

        # Category 3: Product & Sales Pattern Analysis (PSPA)
        with st.expander("📦 Product & Sales Pattern Analysis (PSPA)", expanded=False):
            st.info("Analyze product sales patterns, velocity, and attribute-based trends.")

            # PSPA 1: Product Sales Velocity & Inventory Insights
            st.subheader("PSPA 1: Product Sales Velocity & Inventory Insights")
            st.markdown("Analyze how quickly products sell (sales per period) to identify fast-moving and slow-moving items. This requires 'SKU', 'Date', and 'Qty' columns.")
            
            all_cols_pspa1 = df.columns.tolist()
            date_cols_pspa1 = date_cols
            numeric_cols_pspa1 = get_numeric_columns(df)

            sku_col_pspa1 = st.selectbox("Select Product ID/SKU column:", all_cols_pspa1, index=all_cols_pspa1.index('SKU') if 'SKU' in all_cols_pspa1 else 0, key="pspa1_sku")
            date_col_pspa1 = st.selectbox("Select Date column:", date_cols_pspa1 if date_cols_pspa1 else all_cols_pspa1, index=date_cols_pspa1.index('Date') if 'Date' in date_cols_pspa1 else 0, key="pspa1_date")
            qty_col_pspa1 = st.selectbox("Select Quantity Sold column:", numeric_cols_pspa1, index=numeric_cols_pspa1.index('Qty') if 'Qty' in numeric_cols_pspa1 else 0, key="pspa1_qty")
            
            time_period_pspa1 = st.selectbox("Aggregation period for velocity:", ["D", "W", "M"], index=1, format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x], key="pspa1_period")
            top_n_pspa1 = st.slider("Number of top/bottom products to show:", 5, 20, 10, key="pspa1_top_n")

            if st.button("🏃‍♂️ Analyze Product Sales Velocity", key="pspa1_run"):
                if not all([sku_col_pspa1, date_col_pspa1, qty_col_pspa1]):
                    st.warning("Please select SKU, Date, and Quantity columns.")
                else:
                    try:
                        pspa1_df = df[[sku_col_pspa1, date_col_pspa1, qty_col_pspa1]].copy()
                        pspa1_df[date_col_pspa1] = pd.to_datetime(pspa1_df[date_col_pspa1], errors='coerce')
                        pspa1_df = pspa1_df.dropna()

                        if pspa1_df.empty:
                            st.warning("No data available after filtering for sales velocity analysis.")
                        else:
                            # Calculate total observation period for each product
                            product_days = pspa1_df.groupby(sku_col_pspa1)[date_col_pspa1].agg(['min', 'max'])
                            product_days['duration_days'] = (product_days['max'] - product_days['min']).dt.days + 1 # +1 to include start day
                            
                            # Calculate total quantity sold
                            product_qty = pspa1_df.groupby(sku_col_pspa1)[qty_col_pspa1].sum()
                            
                            velocity_df = pd.concat([product_qty, product_days], axis=1)
                            velocity_df = velocity_df.rename(columns={qty_col_pspa1: 'TotalQtySold'})
                            
                            # Calculate velocity based on selected period
                            if time_period_pspa1 == "D":
                                velocity_df['SalesVelocity'] = velocity_df['TotalQtySold'] / velocity_df['duration_days']
                                velocity_unit = "per day"
                            elif time_period_pspa1 == "W":
                                velocity_df['SalesVelocity'] = velocity_df['TotalQtySold'] / (velocity_df['duration_days'] / 7)
                                velocity_unit = "per week"
                            elif time_period_pspa1 == "M":
                                velocity_df['SalesVelocity'] = velocity_df['TotalQtySold'] / (velocity_df['duration_days'] / 30.44) # Avg days per month
                                velocity_unit = "per month"
                            
                            velocity_df = velocity_df.replace([np.inf, -np.inf], 0).fillna(0) # Handle division by zero if duration is 0

                            st.markdown(f"##### Product Sales Velocity ({velocity_unit})") # Changed subheader level
                            st.markdown(f"###### Top {top_n_pspa1} Fast-Moving Products") # Changed subheader level
                            st.dataframe(velocity_df.sort_values(by='SalesVelocity', ascending=False).head(top_n_pspa1))
                            
                            st.markdown(f"###### Top {top_n_pspa1} Slow-Moving Products (Sales Velocity > 0)") # Changed subheader level
                            st.dataframe(velocity_df[velocity_df['SalesVelocity'] > 0].sort_values(by='SalesVelocity', ascending=True).head(top_n_pspa1))
                            
                            # Plot distribution of sales velocity
                            fig_pspa1_vel, ax_pspa1_vel = plt.subplots()
                            sns.histplot(velocity_df[velocity_df['SalesVelocity'] > 0]['SalesVelocity'], kde=True, ax=ax_pspa1_vel, bins=30)
                            ax_pspa1_vel.set_title(f"Distribution of Sales Velocity ({velocity_unit})")
                            ax_pspa1_vel.set_xlabel(f"Sales Velocity ({velocity_unit})")
                            st.pyplot(fig_pspa1_vel)

                    except Exception as e:
                        st.error(f"Error during sales velocity analysis: {e}")
            st.markdown("---")

            # PSPA 2: Price Elasticity Estimation (Simplified)
            st.subheader("PSPA 2: Price Elasticity Estimation (Simplified)")
            st.markdown("Explore the relationship between price and quantity sold for specific products. This is a simplified analysis and not a formal elasticity calculation. Requires 'SKU', 'Date', 'Amount', 'Qty'.")
            
            all_cols_pspa2 = df.columns.tolist()
            date_cols_pspa2 = date_cols
            numeric_cols_pspa2 = get_numeric_columns(df)

            sku_col_pspa2 = st.selectbox("Select Product ID/SKU column:", all_cols_pspa2, index=all_cols_pspa2.index('SKU') if 'SKU' in all_cols_pspa2 else 0, key="pspa2_sku")
            date_col_pspa2 = st.selectbox("Select Date column:", date_cols_pspa2 if date_cols_pspa2 else all_cols_pspa2, index=date_cols_pspa2.index('Date') if 'Date' in date_cols_pspa2 else 0, key="pspa2_date")
            amount_col_pspa2 = st.selectbox("Select Sales Amount column:", numeric_cols_pspa2, index=numeric_cols_pspa2.index('Amount') if 'Amount' in numeric_cols_pspa2 else 0, key="pspa2_amount")
            qty_col_pspa2 = st.selectbox("Select Quantity Sold column:", numeric_cols_pspa2, index=numeric_cols_pspa2.index('Qty') if 'Qty' in numeric_cols_pspa2 else 0, key="pspa2_qty")
            
            available_skus_pspa2 = df[sku_col_pspa2].dropna().unique()
            selected_sku_pspa2 = st.selectbox("Select a specific SKU to analyze:", available_skus_pspa2, key="pspa2_selected_sku")

            if st.button("📈 Analyze Price-Quantity Relationship", key="pspa2_run"):
                if not all([sku_col_pspa2, date_col_pspa2, amount_col_pspa2, qty_col_pspa2, selected_sku_pspa2]):
                    st.warning("Please select all required columns and a specific SKU.")
                else:
                    try:
                        pspa2_df = df[df[sku_col_pspa2] == selected_sku_pspa2].copy()
                        pspa2_df = pspa2_df[[date_col_pspa2, amount_col_pspa2, qty_col_pspa2]].dropna()
                        
                        if pspa2_df.empty or len(pspa2_df) < 2:
                            st.warning(f"Not enough data for SKU '{selected_sku_pspa2}' to analyze price-quantity relationship.")
                        else:
                            pspa2_df['PricePerUnit'] = pspa2_df[amount_col_pspa2] / pspa2_df[qty_col_pspa2].replace(0, np.nan)
                            pspa2_df = pspa2_df.dropna(subset=['PricePerUnit'])
                            
                            if pspa2_df.empty or len(pspa2_df) < 2:
                                st.warning(f"Not enough data for SKU '{selected_sku_pspa2}' after calculating price per unit.")
                            else:
                                st.markdown(f"##### Price vs. Quantity for SKU: {selected_sku_pspa2}") # Changed subheader level
                                
                                fig_pspa2, ax_pspa2 = plt.subplots()
                                sns.scatterplot(data=pspa2_df, x='PricePerUnit', y=qty_col_pspa2, ax=ax_pspa2)
                                ax_pspa2.set_title(f"Price per Unit vs. Quantity Sold for {selected_sku_pspa2}")
                                ax_pspa2.set_xlabel("Price Per Unit")
                                ax_pspa2.set_ylabel(f"Quantity Sold ({qty_col_pspa2})")
                                st.pyplot(fig_pspa2)

                                st.write("Data points used for the plot:")
                                st.dataframe(pspa2_df[['PricePerUnit', qty_col_pspa2]].head(20))

                                if len(pspa2_df['PricePerUnit'].unique()) > 1 and len(pspa2_df) > 2: # Need variation in price for regression
                                    X_pspa2 = pspa2_df[['PricePerUnit']]
                                    y_pspa2 = pspa2_df[qty_col_pspa2]
                                    model_pspa2 = LinearRegression()
                                    model_pspa2.fit(X_pspa2, y_pspa2)
                                    st.markdown("###### Simplified Linear Regression (Quantity ~ Price)") # Changed subheader level
                                    st.write(f"Coefficient (slope) for PricePerUnit: {model_pspa2.coef_[0]:.2f}")
                                    st.write(f"Intercept: {model_pspa2.intercept_:.2f}")
                                    st.caption("Note: A negative coefficient suggests that as price increases, quantity sold tends to decrease. This is a very simplified model.")
                                else:
                                    st.info("Not enough variation in price or data points for a meaningful linear regression.")
                    except Exception as e:
                        st.error(f"Error during price-quantity analysis: {e}")
            st.markdown("---")

            # PSPA 3: Sales Trend by Product Attribute
            st.subheader("PSPA 3: Sales Trend by Product Attribute")
            st.markdown("Analyze sales trends (Amount or Quantity) over time, grouped by a selected product attribute (e.g., 'Size', 'Style', 'Category').")

            all_cols_pspa3 = df.columns.tolist()
            date_cols_pspa3 = date_cols
            numeric_cols_pspa3 = get_numeric_columns(df)
            categorical_cols_pspa3 = get_categorical_columns(df, nunique_threshold=50) # Allow more unique values for attributes

            date_col_pspa3 = st.selectbox("Select Date column:", date_cols_pspa3 if date_cols_pspa3 else all_cols_pspa3, index=date_cols_pspa3.index('Date') if 'Date' in date_cols_pspa3 else 0, key="pspa3_date")
            value_col_pspa3 = st.selectbox("Select Value column for trend (Amount or Qty):", numeric_cols_pspa3, index=numeric_cols_pspa3.index('Amount') if 'Amount' in numeric_cols_pspa3 else (numeric_cols_pspa3.index('Qty') if 'Qty' in numeric_cols_pspa3 else 0), key="pspa3_value")
            attribute_col_pspa3 = st.selectbox("Select Product Attribute column (Categorical):", [None] + categorical_cols_pspa3, index=0, key="pspa3_attribute")
            
            aggregation_freq_pspa3 = st.selectbox("Aggregate trend by:", ["W", "M", "Q"], index=1, format_func=lambda x: {"W":"Weekly", "M":"Monthly", "Q":"Quarterly"}[x], key="pspa3_freq")

            if st.button("📊 Analyze Sales Trend by Attribute", key="pspa3_run"):
                if not all([date_col_pspa3, value_col_pspa3, attribute_col_pspa3]):
                    st.warning("Please select Date, Value, and Attribute columns.")
                else:
                    try:
                        pspa3_df = df[[date_col_pspa3, value_col_pspa3, attribute_col_pspa3]].copy()
                        pspa3_df[date_col_pspa3] = pd.to_datetime(pspa3_df[date_col_pspa3], errors='coerce')
                        pspa3_df = pspa3_df.dropna()

                        if pspa3_df.empty:
                            st.warning("No data available after filtering for attribute trend analysis.")
                        else:
                            trend_data_pspa3 = pspa3_df.groupby([pd.Grouper(key=date_col_pspa3, freq=aggregation_freq_pspa3), attribute_col_pspa3])[value_col_pspa3].sum().unstack()
                            
                            if trend_data_pspa3.empty:
                                st.warning(f"No trend data to display for attribute '{attribute_col_pspa3}'.")
                            else:
                                st.markdown(f"##### Sales Trend of '{value_col_pspa3}' by '{attribute_col_pspa3}' ({aggregation_freq_pspa3} Aggregation)") # Changed subheader level
                                st.line_chart(trend_data_pspa3.fillna(0)) # Fill NaNs for plotting if some attributes don't appear in all periods
                                
                                st.write("Trend Data Table (Top 20 rows):")
                                st.dataframe(trend_data_pspa3.head(20))
                    except Exception as e:
                        st.error(f"Error during sales trend by attribute analysis: {e}")

            st.markdown("---")
            # PSPA 4: Market Basket Analysis (Product Co-purchase)
            st.subheader("PSPA 4: Market Basket Analysis (Product Co-purchase)")
            st.markdown("Discover which specific products (SKUs/ASINs) are frequently bought together. This uses the Apriori algorithm for association rule mining.")

            all_cols_pspa4 = df.columns.tolist()

            order_id_col_pspa4 = st.selectbox("Select Order ID column:", all_cols_pspa4, index=all_cols_pspa4.index('Order ID') if 'Order ID' in all_cols_pspa4 else 0, key="pspa4_order_id")
            item_id_col_pspa4 = st.selectbox("Select Product/Item ID column (e.g., SKU, ASIN):", all_cols_pspa4, index=all_cols_pspa4.index('SKU') if 'SKU' in all_cols_pspa4 else (all_cols_pspa4.index('ASIN') if 'ASIN' in all_cols_pspa4 else 0), key="pspa4_item_id")
            
            min_support_pspa4 = st.slider("Minimum Support for itemsets:", 0.001, 0.1, 0.01, 0.001, format="%.3f", key="pspa4_min_support", help="Minimum proportion of transactions an itemset must appear in.")
            min_confidence_pspa4 = st.slider("Minimum Confidence for rules:", 0.1, 1.0, 0.2, 0.05, key="pspa4_min_confidence", help="Likelihood that item Y is purchased when item X is purchased (for rule X->Y).")

            if st.button("🧺 Run Market Basket Analysis", key="pspa4_run"):
                if not order_id_col_pspa4 or not item_id_col_pspa4:
                    st.warning("Please select both Order ID and Item ID columns.")
                else:
                    try:
                        mba_df = df[[order_id_col_pspa4, item_id_col_pspa4]].copy().dropna()
                        if mba_df.empty or mba_df[order_id_col_pspa4].nunique() < 2 or mba_df[item_id_col_pspa4].nunique() < 2:
                            st.warning("Not enough data or distinct orders/items for Market Basket Analysis.")
                        else:
                            # Create a list of lists, where each inner list contains items in an order
                            transactions = mba_df.groupby(order_id_col_pspa4)[item_id_col_pspa4].apply(lambda x: list(set(x))).tolist() # Use set to avoid duplicate items within same order if data has them
                            transactions = [t for t in transactions if len(t) > 1] # Only consider transactions with more than one item for co-purchase

                            if not transactions:
                                st.warning("No transactions found with multiple items to analyze for co-purchase.")
                            else:
                                te = TransactionEncoder()
                                te_ary = te.fit(transactions).transform(transactions)
                                basket_df = pd.DataFrame(te_ary, columns=te.columns_)

                                frequent_itemsets = apriori(basket_df, min_support=min_support_pspa4, use_colnames=True, max_len=4) # Limit max_len for performance
                                
                                if frequent_itemsets.empty:
                                    st.info(f"No frequent itemsets found with minimum support of {min_support_pspa4}. Try lowering the support threshold.")
                                else:
                                    st.markdown("###### Frequent Itemsets")
                                    st.dataframe(frequent_itemsets.sort_values(by="support", ascending=False).head(20))

                                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_pspa4)
                                    if rules.empty:
                                        st.info(f"No association rules found with minimum confidence of {min_confidence_pspa4}. Try lowering the confidence threshold or adjusting support.")
                                    else:
                                        st.markdown("###### Association Rules")
                                        st.dataframe(rules.sort_values(by=["lift", "confidence"], ascending=[False, False]).head(30))
                    except Exception as e:
                        st.error(f"Error during Market Basket Analysis: {e}. Ensure 'mlxtend' library is installed.")

        # Category 4: Geospatial & Fulfillment Insights (GFI)
        with st.expander("🌍 Geospatial & Fulfillment Insights (GFI)", expanded=False):
            st.info("Analyze sales performance related to geography and fulfillment methods.")

            # GFI 1: Shipping Performance Analysis
            st.subheader("GFI 1: Shipping Performance Analysis")
            st.markdown("Analyze sales volume or value by 'ship-service-level' and optionally by region ('ship-state').")

            all_cols_gfi1 = df.columns.tolist()
            numeric_cols_gfi1 = get_numeric_columns(df)
            
            service_level_col_gfi1 = st.selectbox("Select Shipping Service Level column:", all_cols_gfi1, index=all_cols_gfi1.index('ship-service-level') if 'ship-service-level' in all_cols_gfi1 else 0, key="gfi1_service_level")
            value_col_gfi1 = st.selectbox("Select Value column (Amount or Qty):", numeric_cols_gfi1, index=numeric_cols_gfi1.index('Amount') if 'Amount' in numeric_cols_gfi1 else (numeric_cols_gfi1.index('Qty') if 'Qty' in numeric_cols_gfi1 else 0), key="gfi1_value")
            state_col_gfi1 = st.selectbox("Optional: Select State column for regional breakdown:", [None] + all_cols_gfi1, index=([None] + all_cols_gfi1).index('ship-state') if 'ship-state' in all_cols_gfi1 else 0, key="gfi1_state")

            if st.button("🚚 Analyze Shipping Performance", key="gfi1_run"):
                if not all([service_level_col_gfi1, value_col_gfi1]):
                    st.warning("Please select Shipping Service Level and Value columns.")
                else:
                    try:
                        gfi1_df = df.copy()
                        group_by_cols = [service_level_col_gfi1]
                        if state_col_gfi1:
                            group_by_cols.append(state_col_gfi1)
                        
                        gfi1_analysis = gfi1_df.groupby(group_by_cols)[value_col_gfi1].agg(['sum', 'count', 'mean']).reset_index()
                        gfi1_analysis.columns = group_by_cols + [f'Total_{value_col_gfi1}', 'Order_Count', f'Avg_{value_col_gfi1}_per_Order']
                        gfi1_analysis = gfi1_analysis.sort_values(by=f'Total_{value_col_gfi1}', ascending=False)

                        if gfi1_analysis.empty:
                            st.warning("No data for shipping performance analysis.")
                        else:
                            st.markdown(f"##### Shipping Performance by '{service_level_col_gfi1}'") # Changed subheader level
                            st.dataframe(gfi1_analysis.head(20))

                            # Plot total value by service level
                            fig_gfi1_total, ax_gfi1_total = plt.subplots()
                            summary_total_gfi1 = gfi1_df.groupby(service_level_col_gfi1)[value_col_gfi1].sum().sort_values(ascending=False)
                            summary_total_gfi1.plot(kind='bar', ax=ax_gfi1_total)
                            ax_gfi1_total.set_title(f'Total {value_col_gfi1} by {service_level_col_gfi1}')
                            ax_gfi1_total.set_ylabel(f'Total {value_col_gfi1}')
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                            st.pyplot(fig_gfi1_total)

                            if state_col_gfi1:
                                st.markdown(f"###### Top States by Total {value_col_gfi1} (Overall)") # Changed subheader level
                                top_states_gfi1 = gfi1_df.groupby(state_col_gfi1)[value_col_gfi1].sum().nlargest(10)
                                st.bar_chart(top_states_gfi1)

                    except Exception as e:
                        st.error(f"Error during shipping performance analysis: {e}")
            st.markdown("---")

            # GFI 2: Regional Product Preferences
            st.subheader("GFI 2: Regional Product Preferences")
            st.markdown("Identify which products or categories are most popular in different regions (e.g., 'ship-state').")

            all_cols_gfi2 = df.columns.tolist()
            numeric_cols_gfi2 = get_numeric_columns(df)
            
            product_level_gfi2 = st.radio("Analyze by:", ["SKU", "Category"], key="gfi2_product_level")
            product_col_gfi2 = st.selectbox(f"Select {product_level_gfi2} column:", all_cols_gfi2, index=all_cols_gfi2.index(product_level_gfi2) if product_level_gfi2 in all_cols_gfi2 else 0, key="gfi2_product")
            region_col_gfi2 = st.selectbox("Select Region column (e.g., ship-state):", all_cols_gfi2, index=all_cols_gfi2.index('ship-state') if 'ship-state' in all_cols_gfi2 else 0, key="gfi2_region")
            value_col_gfi2 = st.selectbox("Select Value column for preference (Amount or Qty):", numeric_cols_gfi2, index=numeric_cols_gfi2.index('Amount') if 'Amount' in numeric_cols_gfi2 else (numeric_cols_gfi2.index('Qty') if 'Qty' in numeric_cols_gfi2 else 0), key="gfi2_value")
            top_n_gfi2 = st.slider("Number of top products/categories per region to show:", 1, 10, 3, key="gfi2_top_n")

            if st.button("🗺️ Analyze Regional Preferences", key="gfi2_run"):
                if not all([product_col_gfi2, region_col_gfi2, value_col_gfi2]):
                    st.warning("Please select Product/Category, Region, and Value columns.")
                else:
                    try:
                        gfi2_df = df[[product_col_gfi2, region_col_gfi2, value_col_gfi2]].copy().dropna()
                        if gfi2_df.empty:
                            st.warning("No data available for regional preference analysis.")
                        else:
                            regional_summary = gfi2_df.groupby([region_col_gfi2, product_col_gfi2])[value_col_gfi2].sum().reset_index()
                            
                            st.markdown(f"##### Top {top_n_gfi2} {product_level_gfi2}s by {value_col_gfi2} per {region_col_gfi2}") # Changed subheader level
                            
                            top_prefs_display = regional_summary.loc[regional_summary.groupby(region_col_gfi2)[value_col_gfi2].nlargest(top_n_gfi2).reset_index(level=0, drop=True).index]
                            top_prefs_display = top_prefs_display.sort_values(by=[region_col_gfi2, value_col_gfi2], ascending=[True, False])

                            if top_prefs_display.empty:
                                st.info("Could not determine top preferences with current settings.")
                            else:
                                st.dataframe(top_prefs_display)

                                # Optional: Heatmap for top N regions and top N products/categories
                                if regional_summary[region_col_gfi2].nunique() < 30 and regional_summary[product_col_gfi2].nunique() < 50 : # Avoid overly large heatmaps
                                    pivot_table_gfi2 = pd.pivot_table(regional_summary, values=value_col_gfi2, index=product_col_gfi2, columns=region_col_gfi2, aggfunc='sum', fill_value=0)
                                    # Select top N products and regions for a more readable heatmap
                                    top_overall_products = pivot_table_gfi2.sum(axis=1).nlargest(15).index
                                    top_overall_regions = pivot_table_gfi2.sum(axis=0).nlargest(10).index
                                    
                                    if not top_overall_products.empty and not top_overall_regions.empty:
                                        heatmap_data_gfi2 = pivot_table_gfi2.loc[top_overall_products, top_overall_regions]
                                        if not heatmap_data_gfi2.empty:
                                            st.markdown(f"###### Heatmap: {value_col_gfi2} of Top {product_level_gfi2}s vs. Top {region_col_gfi2}s") # Changed subheader level
                                            fig_gfi2_hm, ax_gfi2_hm = plt.subplots(figsize=(12, max(8, len(top_overall_products)*0.5)))
                                            sns.heatmap(heatmap_data_gfi2, annot=False, cmap="YlGnBu", ax=ax_gfi2_hm, fmt=".0f") # Annot can be slow for large heatmaps
                                            plt.xticks(rotation=45, ha="right")
                                            plt.yticks(rotation=0)
                                            plt.tight_layout()
                                            st.pyplot(fig_gfi2_hm)
                                        else:
                                            st.info("Not enough overlapping data for heatmap of top items/regions.")
                                    else:
                                        st.info("Not enough distinct products or regions for a summary heatmap.")
                                else:
                                    st.info("Dataset too large for a full heatmap visualization of regional preferences. Showing table instead.")

                    except Exception as e:
                        st.error(f"Error during regional preference analysis: {e}")

        # Category 5: Financial & Order Metrics (FOM)
        with st.expander("💰 Financial & Order Metrics (FOM)", expanded=False):
            st.info("Analyze key financial and order-related metrics like AOV, discount impact, and fulfillment performance.")

            # FOM 1: Average Order Value (AOV) Trend & Drivers
            st.subheader("FOM 1: Average Order Value (AOV) Trend & Drivers")
            st.markdown("Analyze Average Order Value (AOV) over time and by different segments (e.g., 'Sales Channel', 'B2B', 'Fulfilment'). Requires 'Order ID', 'Amount', 'Date'.")

            all_cols_fom1 = df.columns.tolist()
            date_cols_fom1 = date_cols
            numeric_cols_fom1 = get_numeric_columns(df)
            categorical_cols_fom1 = get_categorical_columns(df)

            order_id_col_fom1 = st.selectbox("Select Order ID column:", all_cols_fom1, index=all_cols_fom1.index('Order ID') if 'Order ID' in all_cols_fom1 else 0, key="fom1_order_id")
            amount_col_fom1 = st.selectbox("Select Order Amount column:", numeric_cols_fom1, index=numeric_cols_fom1.index('Amount') if 'Amount' in numeric_cols_fom1 else 0, key="fom1_amount")
            date_col_fom1 = st.selectbox("Select Date column:", date_cols_fom1 if date_cols_fom1 else all_cols_fom1, index=date_cols_fom1.index('Date') if 'Date' in date_cols_fom1 else 0, key="fom1_date")
            
            aggregation_freq_fom1 = st.selectbox("Aggregate AOV trend by:", ["W", "M", "Q"], index=1, format_func=lambda x: {"W":"Weekly", "M":"Monthly", "Q":"Quarterly"}[x], key="fom1_freq")
            segment_col_fom1 = st.selectbox("Optional: Segment AOV by column:", [None] + categorical_cols_fom1, index=0, key="fom1_segment")

            if st.button("💲 Analyze AOV", key="fom1_run"):
                if not all([order_id_col_fom1, amount_col_fom1, date_col_fom1]):
                    st.warning("Please select Order ID, Amount, and Date columns.")
                else:
                    try:
                        fom1_df = df[[order_id_col_fom1, amount_col_fom1, date_col_fom1]].copy()
                        # If the data is item-level and Order ID can have multiple items, first sum Amount per Order ID.
                        # The sample data has Order ID unique per row, so each row is an "order" in this context.
                        # If Order ID was not unique, we'd do:
                        # order_amounts = fom1_df.groupby([order_id_col_fom1, date_col_fom1])[amount_col_fom1].sum().reset_index()
                        # For this dataset, we assume each row is a distinct order item, and Order ID is the transaction.
                        # If an Order ID can have multiple items, we need to group by Order ID first to get total order amount.
                        # Let's assume the current 'Amount' is per item, and we need to sum it up per 'Order ID' if 'Order ID' is not unique per transaction.
                        # Given the dataset structure, 'Order ID' seems to be unique per row (item).
                        # So, AOV calculation needs to be based on unique orders.
                        
                        fom1_df[date_col_fom1] = pd.to_datetime(fom1_df[date_col_fom1], errors='coerce')
                        fom1_df = fom1_df.dropna(subset=[order_id_col_fom1, amount_col_fom1, date_col_fom1])

                        if fom1_df.empty:
                            st.warning("No data available for AOV analysis.")
                        else:
                            # Calculate AOV: Total Revenue / Number of Unique Orders
                            # For trend, group by time period
                            fom1_df['time_period'] = fom1_df[date_col_fom1].dt.to_period(aggregation_freq_fom1)
                            
                            if segment_col_fom1 and segment_col_fom1 in df.columns:
                                fom1_df[segment_col_fom1] = df[segment_col_fom1] # Add segment column
                                aov_trend_data = fom1_df.groupby(['time_period', segment_col_fom1]).agg(
                                    TotalRevenue=(amount_col_fom1, 'sum'),
                                    UniqueOrders=(order_id_col_fom1, 'nunique')
                                )
                                aov_trend_data['AOV'] = aov_trend_data['TotalRevenue'] / aov_trend_data['UniqueOrders']
                                aov_trend_plot = aov_trend_data['AOV'].unstack()
                                st.markdown(f"##### AOV Trend by '{segment_col_fom1}' ({aggregation_freq_fom1} Aggregation)") # Changed subheader level
                            else:
                                aov_trend_data = fom1_df.groupby('time_period').agg(
                                    TotalRevenue=(amount_col_fom1, 'sum'),
                                    UniqueOrders=(order_id_col_fom1, 'nunique')
                                )
                                aov_trend_data['AOV'] = aov_trend_data['TotalRevenue'] / aov_trend_data['UniqueOrders']
                                aov_trend_plot = aov_trend_data['AOV']
                                st.markdown(f"##### Overall AOV Trend ({aggregation_freq_fom1} Aggregation)") # Changed subheader level

                            aov_trend_plot = aov_trend_plot.fillna(0)
                            if not aov_trend_plot.empty:
                                st.line_chart(aov_trend_plot)
                                st.write("AOV Data (first 20 periods):")
                                st.dataframe(aov_trend_data.head(20))
                            else:
                                st.info("No AOV trend data to display.")
                            
                            # Overall AOV
                            overall_aov = fom1_df[amount_col_fom1].sum() / fom1_df[order_id_col_fom1].nunique()
                            st.metric("Overall Average Order Value (AOV)", f"{overall_aov:,.2f}")

                    except Exception as e:
                        st.error(f"Error during AOV analysis: {e}")
            st.markdown("---")

            # FOM 2: Discount Impact Analysis (using promotion-ids)
            st.subheader("FOM 2: Discount Impact Analysis (using promotion-ids)")
            st.markdown("Analyze the impact of promotions (indicated by non-empty 'promotion-ids') on sales metrics like Average Item Price, Quantity per Order, and Total Revenue. This is an extension of the basic promotion tool.")

            all_cols_fom2 = df.columns.tolist()
            numeric_cols_fom2 = get_numeric_columns(df)

            promo_id_col_fom2 = st.selectbox("Select Promotion IDs column:", all_cols_fom2, index=all_cols_fom2.index('promotion-ids') if 'promotion-ids' in all_cols_fom2 else 0, key="fom2_promo_id")
            amount_col_fom2 = st.selectbox("Select Sales Amount column:", numeric_cols_fom2, index=numeric_cols_fom2.index('Amount') if 'Amount' in numeric_cols_fom2 else 0, key="fom2_amount")
            qty_col_fom2 = st.selectbox("Select Quantity Sold column:", numeric_cols_fom2, index=numeric_cols_fom2.index('Qty') if 'Qty' in numeric_cols_fom2 else 0, key="fom2_qty")
            order_id_col_fom2 = st.selectbox("Select Order ID column:", all_cols_fom2, index=all_cols_fom2.index('Order ID') if 'Order ID' in all_cols_fom2 else 0, key="fom2_order_id")

            if st.button("💸 Analyze Discount Impact", key="fom2_run"):
                if not all([promo_id_col_fom2, amount_col_fom2, qty_col_fom2, order_id_col_fom2]):
                    st.warning("Please select Promotion ID, Amount, Quantity, and Order ID columns.")
                else:
                    try:
                        fom2_df = df[[promo_id_col_fom2, amount_col_fom2, qty_col_fom2, order_id_col_fom2]].copy().dropna(subset=[amount_col_fom2, qty_col_fom2, order_id_col_fom2])
                        fom2_df['HasPromotion'] = ~fom2_df[promo_id_col_fom2].isnull() & (fom2_df[promo_id_col_fom2].astype(str).str.strip() != '')

                        if fom2_df.empty:
                            st.warning("No data available for discount impact analysis.")
                        else:
                            st.markdown("##### Discount Impact Analysis Results") # Changed subheader level
                            
                            # Item-level analysis
                            item_level_summary = fom2_df.groupby('HasPromotion').agg(
                                TotalRevenue=(amount_col_fom2, 'sum'),
                                TotalQuantity=(qty_col_fom2, 'sum'),
                                NumberOfItems=(order_id_col_fom2, 'count'), # Count of line items
                                AverageItemPrice=(amount_col_fom2, 'mean')
                            ).rename(index={True: 'With Promotion', False: 'Without Promotion'})
                            item_level_summary['AverageItemsPerTransactionEstimate'] = item_level_summary['TotalQuantity'] / fom2_df.groupby('HasPromotion')[order_id_col_fom2].nunique()

                            st.markdown("###### Item-Level Performance (Promoted vs. Non-Promoted Items)") # Changed subheader level
                            st.dataframe(item_level_summary)

                            fig_fom2_rev, ax_fom2_rev = plt.subplots()
                            item_level_summary['TotalRevenue'].plot(kind='bar', ax=ax_fom2_rev, title='Total Revenue by Promotion Status')
                            st.pyplot(fig_fom2_rev)

                            # Order-level analysis (AOV for orders with/without any promoted item)
                            # This requires identifying if an order contains ANY promoted item.
                            order_promo_status = fom2_df.groupby(order_id_col_fom2)['HasPromotion'].any().reset_index(name='OrderHasPromotion')
                            order_data_fom2 = fom2_df.groupby(order_id_col_fom2).agg(
                                OrderAmount=(amount_col_fom2, 'sum'),
                                OrderQuantity=(qty_col_fom2, 'sum')
                            ).reset_index()
                            order_data_fom2 = pd.merge(order_data_fom2, order_promo_status, on=order_id_col_fom2)

                            order_level_summary = order_data_fom2.groupby('OrderHasPromotion').agg(
                                TotalRevenue=('OrderAmount', 'sum'),
                                NumberOfOrders=(order_id_col_fom2, 'count'),
                                AverageOrderValue=('OrderAmount', 'mean'),
                                AverageQuantityPerOrder=('OrderQuantity', 'mean')
                            ).rename(index={True: 'Order Contains Promoted Item(s)', False: 'Order Contains No Promoted Items'})
                            
                            st.markdown("###### Order-Level Performance (Orders with vs. without Promoted Items)") # Changed subheader level
                            st.dataframe(order_level_summary)

                            fig_fom2_aov, ax_fom2_aov = plt.subplots()
                            order_level_summary['AverageOrderValue'].plot(kind='bar', ax=ax_fom2_aov, title='Average Order Value by Promotion Status in Order')
                            st.pyplot(fig_fom2_aov)

                    except Exception as e:
                        st.error(f"Error during discount impact analysis: {e}")
            st.markdown("---")

            # FOM 3: Fulfilment Method Deep Dive
            st.subheader("FOM 3: Fulfilment Method Deep Dive")
            st.markdown("Analyze how 'Fulfilment' method (e.g., Amazon, Merchant) impacts sales metrics like average item price, quantity per order, and revenue by category.")

            all_cols_fom3 = df.columns.tolist()
            numeric_cols_fom3 = get_numeric_columns(df)
            categorical_cols_fom3 = get_categorical_columns(df)

            fulfilment_col_fom3 = st.selectbox("Select Fulfilment column:", all_cols_fom3, index=all_cols_fom3.index('Fulfilment') if 'Fulfilment' in all_cols_fom3 else 0, key="fom3_fulfilment")
            amount_col_fom3 = st.selectbox("Select Sales Amount column:", numeric_cols_fom3, index=numeric_cols_fom3.index('Amount') if 'Amount' in numeric_cols_fom3 else 0, key="fom3_amount")
            qty_col_fom3 = st.selectbox("Select Quantity Sold column:", numeric_cols_fom3, index=numeric_cols_fom3.index('Qty') if 'Qty' in numeric_cols_fom3 else 0, key="fom3_qty")
            order_id_col_fom3 = st.selectbox("Select Order ID column:", all_cols_fom3, index=all_cols_fom3.index('Order ID') if 'Order ID' in all_cols_fom3 else 0, key="fom3_order_id")
            category_col_fom3 = st.selectbox("Optional: Select Category column for breakdown:", [None] + categorical_cols_fom3, index=([None] + categorical_cols_fom3).index('Category') if 'Category' in categorical_cols_fom3 else 0, key="fom3_category")

            if st.button("📦 Analyze Fulfilment Method", key="fom3_run"):
                if not all([fulfilment_col_fom3, amount_col_fom3, qty_col_fom3, order_id_col_fom3]):
                    st.warning("Please select Fulfilment, Amount, Quantity, and Order ID columns.")
                else:
                    try:
                        fom3_df = df.copy()
                        
                        st.markdown(f"##### Fulfilment Method Analysis ('{fulfilment_col_fom3}')") # Changed subheader level

                        # Overall summary by fulfilment type
                        overall_summary_fom3 = fom3_df.groupby(fulfilment_col_fom3).agg(
                            TotalRevenue=(amount_col_fom3, 'sum'),
                            TotalQuantity=(qty_col_fom3, 'sum'),
                            NumberOfTransactions=(order_id_col_fom3, 'count'), # Assuming each row is a transaction/item
                            AverageItemPrice=(amount_col_fom3, 'mean'),
                            AverageItemQuantity=(qty_col_fom3, 'mean')
                        ).sort_values(by='TotalRevenue', ascending=False)
                        
                        st.markdown("###### Overall Performance by Fulfilment Method") # Changed subheader level
                        st.dataframe(overall_summary_fom3)

                        fig_fom3_rev, ax_fom3_rev = plt.subplots()
                        overall_summary_fom3['TotalRevenue'].plot(kind='pie', autopct='%1.1f%%', ax=ax_fom3_rev, title=f'Revenue Share by {fulfilment_col_fom3}')
                        st.pyplot(fig_fom3_rev)

                        if category_col_fom3:
                            st.markdown(f"###### Revenue by {fulfilment_col_fom3} and {category_col_fom3}") # Changed subheader level
                            fulfilment_category_rev = pd.pivot_table(fom3_df,
                                                                     values=amount_col_fom3,
                                                                     index=category_col_fom3,
                                                                     columns=fulfilment_col_fom3,
                                                                     aggfunc='sum',
                                                                     fill_value=0)
                            if not fulfilment_category_rev.empty:
                                st.dataframe(fulfilment_category_rev.head(15)) # Show top categories
                                
                                # Stacked bar chart for top N categories
                                top_n_cat_fom3 = 10
                                plot_data_fom3_cat = fulfilment_category_rev.sum(axis=1).nlargest(top_n_cat_fom3).index
                                if not plot_data_fom3_cat.empty:
                                    fig_fom3_cat, ax_fom3_cat = plt.subplots(figsize=(12,7))
                                    fulfilment_category_rev.loc[plot_data_fom3_cat].plot(kind='bar', stacked=True, ax=ax_fom3_cat)
                                    ax_fom3_cat.set_title(f'Top {top_n_cat_fom3} Categories: Revenue by {fulfilment_col_fom3}')
                                    ax_fom3_cat.set_ylabel('Total Revenue')
                                    plt.xticks(rotation=45, ha="right")
                                    plt.tight_layout()
                                    st.pyplot(fig_fom3_cat)
                            else:
                                st.info(f"No data for {fulfilment_col_fom3} vs {category_col_fom3} breakdown.")
                    except Exception as e:
                        st.error(f"Error during fulfilment method analysis: {e}")
            st.markdown("---")

            # FOM 4: Payment Method Analysis
            st.subheader("FOM 4: Payment Method Analysis")
            st.markdown("Analyze sales performance across different payment methods. Requires a column identifying payment type, along with sales amount and order ID.")

            all_cols_fom4 = df.columns.tolist()
            numeric_cols_fom4 = get_numeric_columns(df)
            categorical_cols_fom4 = get_categorical_columns(df)
            date_cols_fom4 = date_cols

            payment_col_fom4 = st.selectbox("Select Payment Method column (Categorical):", [None] + categorical_cols_fom4, key="fom4_payment_method")
            amount_col_fom4 = st.selectbox("Select Sales Amount column:", numeric_cols_fom4, index=numeric_cols_fom4.index('Amount') if 'Amount' in numeric_cols_fom4 else 0, key="fom4_amount")
            order_id_col_fom4 = st.selectbox("Select Order ID column:", all_cols_fom4, index=all_cols_fom4.index('Order ID') if 'Order ID' in all_cols_fom4 else 0, key="fom4_order_id")
            date_col_fom4 = st.selectbox("Select Date column (for trends):", date_cols_fom4 if date_cols_fom4 else all_cols_fom4, index=date_cols_fom4.index('Date') if 'Date' in date_cols_fom4 else 0, key="fom4_date")
            aggregation_freq_fom4 = st.selectbox("Aggregate trend by:", ["W", "M", "Q"], index=1, format_func=lambda x: {"W":"Weekly", "M":"Monthly", "Q":"Quarterly"}[x], key="fom4_freq")

            if st.button("💳 Analyze Payment Methods", key="fom4_run"):
                if not all([payment_col_fom4, amount_col_fom4, order_id_col_fom4, date_col_fom4]):
                    st.warning("Please select Payment Method, Amount, Order ID, and Date columns.")
                elif payment_col_fom4 not in df.columns:
                    st.warning(f"Selected Payment Method column '{payment_col_fom4}' not found in the dataset.")
                else:
                    try:
                        fom4_df = df[[payment_col_fom4, amount_col_fom4, order_id_col_fom4, date_col_fom4]].copy()
                        fom4_df[date_col_fom4] = pd.to_datetime(fom4_df[date_col_fom4], errors='coerce')
                        fom4_df = fom4_df.dropna()

                        if fom4_df.empty:
                            st.warning("No data available for payment method analysis after filtering.")
                        else:
                            st.markdown(f"##### Payment Method Performance ('{payment_col_fom4}')")

                            payment_summary = fom4_df.groupby(payment_col_fom4).agg(
                                TotalRevenue=(amount_col_fom4, 'sum'),
                                NumberOfOrders=(order_id_col_fom4, 'nunique'),
                            ).sort_values(by='TotalRevenue', ascending=False)
                            payment_summary['AverageOrderValue'] = payment_summary['TotalRevenue'] / payment_summary['NumberOfOrders']
                            
                            st.markdown("###### Overall Performance by Payment Method")
                            st.dataframe(payment_summary)

                            fig_fom4_rev, ax_fom4_rev = plt.subplots()
                            payment_summary['TotalRevenue'].plot(kind='bar', ax=ax_fom4_rev)
                            ax_fom4_rev.set_title(f'Total Revenue by {payment_col_fom4}')
                            ax_fom4_rev.set_ylabel('Total Revenue')
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                            st.pyplot(fig_fom4_rev)

                            st.markdown(f"###### AOV Trend by {payment_col_fom4}")
                            fom4_df['time_period'] = fom4_df[date_col_fom4].dt.to_period(aggregation_freq_fom4)
                            aov_trend_fom4 = fom4_df.groupby(['time_period', payment_col_fom4]).agg(
                                TotalRevenue=(amount_col_fom4, 'sum'),
                                UniqueOrders=(order_id_col_fom4, 'nunique')
                            )
                            aov_trend_fom4['AOV'] = aov_trend_fom4['TotalRevenue'] / aov_trend_fom4['UniqueOrders']
                            aov_trend_plot_fom4 = aov_trend_fom4['AOV'].unstack().fillna(0)
                            if not aov_trend_plot_fom4.empty:
                                st.line_chart(aov_trend_plot_fom4)
                            else:
                                st.info("No AOV trend data to display for payment methods.")

                    except Exception as e:
                        st.error(f"Error during Payment Method Analysis: {e}")
            st.markdown("---")

            # FOM 5: Customer Lifetime Value (CLV) Estimation (Simplified)
            st.subheader("FOM 5: Customer Lifetime Value (CLV) Estimation (Simplified)")
            st.markdown("Estimate CLV using historical purchase data. Requires a Customer ID (or proxy), Order Date, and Amount.")
            
            all_cols_fom5 = df.columns.tolist()
            numeric_cols_fom5 = get_numeric_columns(df)
            date_cols_fom5 = date_cols

            # Using 'Order ID' as a proxy for Customer ID if no dedicated Customer ID exists.
            # This is a simplification; true CLV needs a stable Customer ID.
            customer_id_col_fom5 = st.selectbox("Select Customer ID column (or 'Order ID' as proxy):", all_cols_fom5, index=all_cols_fom5.index('Order ID') if 'Order ID' in all_cols_fom5 else 0, key="fom5_customer_id")
            date_col_fom5 = st.selectbox("Select Order Date column:", date_cols_fom5 if date_cols_fom5 else all_cols_fom5, index=date_cols_fom5.index('Date') if 'Date' in date_cols_fom5 else 0, key="fom5_date")
            amount_col_fom5 = st.selectbox("Select Sales Amount column:", numeric_cols_fom5, index=numeric_cols_fom5.index('Amount') if 'Amount' in numeric_cols_fom5 else 0, key="fom5_amount")

            if st.button("💖 Estimate Simplified CLV", key="fom5_run"):
                if not all([customer_id_col_fom5, date_col_fom5, amount_col_fom5]):
                    st.warning("Please select Customer ID, Order Date, and Amount columns.")
                else:
                    try:
                        fom5_df = df[[customer_id_col_fom5, date_col_fom5, amount_col_fom5]].copy()
                        fom5_df[date_col_fom5] = pd.to_datetime(fom5_df[date_col_fom5], errors='coerce')
                        fom5_df = fom5_df.dropna()

                        if fom5_df.empty:
                            st.warning("No data available for CLV estimation after filtering.")
                        else:
                            st.markdown("##### Simplified CLV Estimation Results")
                            
                            customer_summary = fom5_df.groupby(customer_id_col_fom5).agg(
                                TotalSpent=(amount_col_fom5, 'sum'),
                                FirstPurchaseDate=(date_col_fom5, 'min'),
                                LastPurchaseDate=(date_col_fom5, 'max'),
                                NumberOfPurchases=(date_col_fom5, 'count') # Count of transactions/items
                            )
                            customer_summary['CustomerLifespanDays'] = (customer_summary['LastPurchaseDate'] - customer_summary['FirstPurchaseDate']).dt.days
                            # For customers with only one purchase, lifespan is 0. Consider adding 1 day or specific handling.
                            customer_summary.loc[customer_summary['NumberOfPurchases'] == 1, 'CustomerLifespanDays'] = 1

                            avg_clv = customer_summary['TotalSpent'].mean()
                            avg_purchase_freq = customer_summary['NumberOfPurchases'].mean()
                            avg_lifespan_days = customer_summary[customer_summary['CustomerLifespanDays'] > 0]['CustomerLifespanDays'].mean() # Exclude 0-day lifespans for avg

                            st.metric("Average Simplified CLV (Total Spend per Customer)", f"{avg_clv:,.2f}")
                            st.metric("Average Purchase Frequency (Transactions per Customer)", f"{avg_purchase_freq:,.2f}")
                            st.metric("Average Customer Lifespan (Days with Activity)", f"{avg_lifespan_days:,.2f} days" if pd.notna(avg_lifespan_days) else "N/A (mostly single purchases)")

                            st.markdown("###### Distribution of Total Spend per Customer")
                            fig_fom5_spend, ax_fom5_spend = plt.subplots()
                            sns.histplot(customer_summary['TotalSpent'], kde=True, ax=ax_fom5_spend, bins=30)
                            ax_fom5_spend.set_title("Distribution of Total Spend per Customer")
                            ax_fom5_spend.set_xlabel("Total Spend")
                            st.pyplot(fig_fom5_spend)

                            st.markdown("###### Top Customers by Total Spend")
                            st.dataframe(customer_summary.sort_values(by='TotalSpent', ascending=False).head(10))
                            st.caption(f"Note: If '{customer_id_col_fom5}' is 'Order ID', this represents spend per order, not per unique customer.")

                    except Exception as e:
                        st.error(f"Error during Simplified CLV Estimation: {e}")
            st.markdown("---")

            # FOM 6: COGS & Profit Margin Analysis (Conceptual)
            st.subheader("FOM 6: COGS & Profit Margin Analysis (Conceptual)")
            st.markdown("Estimate gross profit and margin. Requires 'Amount' and ideally a 'COGS' column or an average margin input.")

            all_cols_fom6 = df.columns.tolist()
            numeric_cols_fom6 = get_numeric_columns(df)
            
            amount_col_fom6 = st.selectbox("Select Sales Amount column:", numeric_cols_fom6, index=numeric_cols_fom6.index('Amount') if 'Amount' in numeric_cols_fom6 else 0, key="fom6_amount")
            sku_col_fom6 = st.selectbox("Select Product ID/SKU column:", all_cols_fom6, index=all_cols_fom6.index('SKU') if 'SKU' in all_cols_fom6 else 0, key="fom6_sku")
            
            cogs_option_fom6 = st.radio("COGS Input Method:", ["Select COGS Column (if available)", "Input Average Gross Margin %"], key="fom6_cogs_option")
            cogs_col_fom6 = None
            avg_margin_fom6 = None

            if cogs_option_fom6 == "Select COGS Column (if available)":
                cogs_col_fom6 = st.selectbox("Select COGS column:", [None] + numeric_cols_fom6, key="fom6_cogs_col")
            else:
                avg_margin_fom6 = st.slider("Average Gross Margin (%) for estimation:", 0, 100, 30, key="fom6_avg_margin")

            if st.button("💹 Analyze Profitability (Conceptual)", key="fom6_run"):
                if not amount_col_fom6 or not sku_col_fom6:
                    st.warning("Please select Amount and SKU columns.")
                elif cogs_option_fom6 == "Select COGS Column (if available)" and not cogs_col_fom6:
                    st.warning("Please select a COGS column or choose to input average margin.")
                else:
                    try:
                        fom6_df = df[[sku_col_fom6, amount_col_fom6]].copy()
                        if cogs_col_fom6 and cogs_col_fom6 in df.columns:
                            fom6_df['COGS'] = df[cogs_col_fom6]
                        elif avg_margin_fom6 is not None:
                            fom6_df['COGS'] = fom6_df[amount_col_fom6] * (1 - (avg_margin_fom6 / 100.0))
                        else:
                            st.error("COGS data or average margin not provided.")
                            st.stop()
                        
                        fom6_df = fom6_df.dropna(subset=[amount_col_fom6, 'COGS'])
                        fom6_df['GrossProfit'] = fom6_df[amount_col_fom6] - fom6_df['COGS']
                        fom6_df['GrossMarginPercentage'] = (fom6_df['GrossProfit'] / fom6_df[amount_col_fom6].replace(0, np.nan)) * 100
                        fom6_df = fom6_df.dropna(subset=['GrossMarginPercentage'])

                        if fom6_df.empty:
                            st.warning("No data available for profitability analysis after calculations.")
                        else:
                            st.markdown("##### Conceptual Profitability Analysis Results")
                            
                            overall_gross_profit = fom6_df['GrossProfit'].sum()
                            overall_revenue = fom6_df[amount_col_fom6].sum()
                            overall_margin = (overall_gross_profit / overall_revenue) * 100 if overall_revenue else 0

                            st.metric("Estimated Total Gross Profit", f"{overall_gross_profit:,.2f}")
                            st.metric("Estimated Overall Gross Margin", f"{overall_margin:,.2f}%")

                            st.markdown(f"###### Top 10 Products by Estimated Gross Profit (based on '{sku_col_fom6}')")
                            product_profit = fom6_df.groupby(sku_col_fom6).agg(
                                TotalRevenue=(amount_col_fom6, 'sum'),
                                TotalCOGS=('COGS', 'sum'),
                                TotalGrossProfit=('GrossProfit', 'sum')
                            ).sort_values(by='TotalGrossProfit', ascending=False)
                            product_profit['AverageMargin'] = (product_profit['TotalGrossProfit'] / product_profit['TotalRevenue'].replace(0,np.nan)) * 100
                            st.dataframe(product_profit.head(10))

                            fig_fom6_profit, ax_fom6_profit = plt.subplots()
                            product_profit['TotalGrossProfit'].head(10).plot(kind='bar', ax=ax_fom6_profit)
                            ax_fom6_profit.set_title(f'Top 10 {sku_col_fom6} by Estimated Gross Profit')
                            ax_fom6_profit.set_ylabel('Estimated Gross Profit')
                            plt.xticks(rotation=45, ha="right")
                            plt.tight_layout()
                            st.pyplot(fig_fom6_profit)

                    except Exception as e:
                        st.error(f"Error during Conceptual Profitability Analysis: {e}")
            st.markdown("---")

            # FOM 7: Order Cancellation Rate Analysis
            st.subheader("FOM 7: Order Cancellation Rate Analysis")
            st.markdown("Analyze the rate of cancelled orders. Requires 'Order ID', 'Status', and 'Date' columns. Define which status values indicate a cancellation.")

            all_cols_fom7 = df.columns.tolist()
            date_cols_fom7 = date_cols

            order_id_col_fom7 = st.selectbox("Select Order ID column:", all_cols_fom7, index=all_cols_fom7.index('Order ID') if 'Order ID' in all_cols_fom7 else 0, key="fom7_order_id")
            status_col_fom7 = st.selectbox("Select Order Status column:", all_cols_fom7, index=all_cols_fom7.index('Status') if 'Status' in all_cols_fom7 else 0, key="fom7_status")
            date_col_fom7 = st.selectbox("Select Date column:", date_cols_fom7 if date_cols_fom7 else all_cols_fom7, index=date_cols_fom7.index('Date') if 'Date' in date_cols_fom7 else 0, key="fom7_date")
            
            # Dynamically get status values for multiselect
            available_statuses_fom7 = df[status_col_fom7].dropna().unique().tolist() if status_col_fom7 and status_col_fom7 in df.columns else []
            default_cancelled_statuses = [s for s in ['Cancelled', 'Shipped - Returned to Seller'] if s in available_statuses_fom7]
            cancelled_statuses_fom7 = st.multiselect("Select status values that indicate cancellation:", available_statuses_fom7, default=default_cancelled_statuses, key="fom7_cancelled_statuses")
            aggregation_freq_fom7 = st.selectbox("Aggregate cancellation trend by:", ["D", "W", "M"], index=1, format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x], key="fom7_freq")

            if st.button("🚫 Analyze Cancellation Rate", key="fom7_run"):
                if not all([order_id_col_fom7, status_col_fom7, date_col_fom7]):
                    st.warning("Please select Order ID, Status, and Date columns.")
                elif not cancelled_statuses_fom7:
                    st.warning("Please select at least one status value that indicates cancellation.")
                else:
                    try:
                        fom7_df = df[[order_id_col_fom7, status_col_fom7, date_col_fom7]].copy()
                        fom7_df[date_col_fom7] = pd.to_datetime(fom7_df[date_col_fom7], errors='coerce')
                        fom7_df = fom7_df.dropna()

                        if fom7_df.empty:
                            st.warning("No data available for cancellation analysis after filtering.")
                        else:
                            fom7_df['IsCancelled'] = fom7_df[status_col_fom7].isin(cancelled_statuses_fom7)
                            
                            # Assuming each row with a unique Order ID is one order. If Order ID can repeat for items, this needs adjustment.
                            # For this dataset, Order ID is unique per row, so we treat each row as an order for status.
                            total_orders = fom7_df[order_id_col_fom7].nunique()
                            cancelled_orders = fom7_df[fom7_df['IsCancelled']][order_id_col_fom7].nunique()
                            cancellation_rate = (cancelled_orders / total_orders) * 100 if total_orders > 0 else 0

                            st.markdown("##### Order Cancellation Analysis Results")
                            st.metric("Overall Cancellation Rate", f"{cancellation_rate:.2f}% ({cancelled_orders}/{total_orders} orders)")

                            st.markdown("###### Cancellation Rate Over Time")
                            fom7_df['time_period'] = fom7_df[date_col_fom7].dt.to_period(aggregation_freq_fom7)
                            cancellation_trend = fom7_df.groupby('time_period').agg(
                                TotalOrders=(order_id_col_fom7, 'nunique'),
                                CancelledOrders=('IsCancelled', lambda x: x.astype(int).sum()) # Sum of True values
                            )
                            cancellation_trend['CancellationRate'] = (cancellation_trend['CancelledOrders'] / cancellation_trend['TotalOrders'].replace(0,np.nan)) * 100
                            
                            if not cancellation_trend.empty:
                                st.line_chart(cancellation_trend['CancellationRate'].fillna(0))
                                st.dataframe(cancellation_trend.head(10))
                            else:
                                st.info("No data for cancellation rate trend.")

                            st.markdown(f"###### Breakdown by Status (within selected '{status_col_fom7}')")
                            status_counts = fom7_df[status_col_fom7].value_counts()
                            st.bar_chart(status_counts)

                    except Exception as e:
                        st.error(f"Error during Order Cancellation Rate Analysis: {e}")
            st.markdown("---")

            # FOM 8: Repeat Purchase Rate & Cohort Analysis (Simplified)
            st.subheader("FOM 8: Repeat Purchase Rate & Cohort Analysis (Simplified)")
            st.markdown("Analyze customer repeat purchase behavior. Requires Customer ID (or proxy) and Order Date.")

            all_cols_fom8 = df.columns.tolist()
            date_cols_fom8 = date_cols

            customer_id_col_fom8 = st.selectbox("Select Customer ID column (or 'Order ID' as proxy):", all_cols_fom8, index=all_cols_fom8.index('Order ID') if 'Order ID' in all_cols_fom8 else 0, key="fom8_customer_id")
            date_col_fom8 = st.selectbox("Select Order Date column:", date_cols_fom8 if date_cols_fom8 else all_cols_fom8, index=date_cols_fom8.index('Date') if 'Date' in date_cols_fom8 else 0, key="fom8_date")

            if st.button("🔁 Analyze Repeat Purchases", key="fom8_run"):
                if not all([customer_id_col_fom8, date_col_fom8]):
                    st.warning("Please select Customer ID and Order Date columns.")
                else:
                    try:
                        fom8_df = df[[customer_id_col_fom8, date_col_fom8]].copy()
                        fom8_df[date_col_fom8] = pd.to_datetime(fom8_df[date_col_fom8], errors='coerce')
                        fom8_df = fom8_df.dropna().sort_values(by=[customer_id_col_fom8, date_col_fom8])

                        if fom8_df.empty:
                            st.warning("No data available for repeat purchase analysis.")
                        else:
                            st.markdown("##### Simplified Repeat Purchase & Cohort Analysis")

                            # Calculate number of purchases per customer
                            purchase_counts = fom8_df.groupby(customer_id_col_fom8).size() # .size() gives count of rows (purchases)
                            
                            total_customers = purchase_counts.count()
                            repeat_customers = purchase_counts[purchase_counts > 1].count()
                            repeat_purchase_rate = (repeat_customers / total_customers) * 100 if total_customers > 0 else 0

                            st.metric("Total Unique Customers/Entities", total_customers)
                            st.metric("Customers with >1 Purchase", repeat_customers)
                            st.metric("Overall Repeat Purchase Rate", f"{repeat_purchase_rate:.2f}%")

                            st.markdown("###### Distribution of Number of Purchases per Customer")
                            fig_fom8_counts, ax_fom8_counts = plt.subplots()
                            # Cap display for readability if many single purchases
                            display_purchase_counts = purchase_counts[purchase_counts <= purchase_counts.quantile(0.95) if purchase_counts.quantile(0.95) > 1 else 10] 
                            sns.histplot(display_purchase_counts, discrete=True, ax=ax_fom8_counts)
                            ax_fom8_counts.set_title("Distribution of Purchase Counts (capped for display)")
                            ax_fom8_counts.set_xlabel("Number of Purchases")
                            st.pyplot(fig_fom8_counts)

                            # Simplified Cohort: First purchase month
                            fom8_df['PurchaseMonth'] = fom8_df[date_col_fom8].dt.to_period('M')
                            fom8_df['CohortMonth'] = fom8_df.groupby(customer_id_col_fom8)['PurchaseMonth'].transform('min')
                            
                            def get_cohort_period(df_cohort, event_month_col='PurchaseMonth', cohort_month_col='CohortMonth'):
                                df_cohort['CohortPeriod'] = (df_cohort[event_month_col].dt.year - df_cohort[cohort_month_col].dt.year) * 12 + \
                                                          (df_cohort[event_month_col].dt.month - df_cohort[cohort_month_col].dt.month)
                                return df_cohort

                            fom8_df = get_cohort_period(fom8_df)
                            
                            cohort_data = fom8_df.groupby(['CohortMonth', 'CohortPeriod'])[customer_id_col_fom8].nunique().reset_index()
                            cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortPeriod', values=customer_id_col_fom8)
                            
                            cohort_sizes = cohort_pivot.iloc[:, 0]
                            retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

                            st.markdown("###### Simplified Monthly Cohort Retention (%)")
                            if not retention_matrix.empty:
                                fig_fom8_cohort, ax_fom8_cohort = plt.subplots(figsize=(10, max(6, len(retention_matrix)*0.4)))
                                sns.heatmap(retention_matrix, annot=True, fmt='.1f', cmap='viridis', ax=ax_fom8_cohort)
                                ax_fom8_cohort.set_title('Monthly Cohort Retention (Customers Active in Period N after First Purchase)')
                                ax_fom8_cohort.set_xlabel('Months Since First Purchase (Period)')
                                ax_fom8_cohort.set_ylabel('Cohort (First Purchase Month)')
                                plt.tight_layout()
                                st.pyplot(fig_fom8_cohort)
                                st.caption(f"Note: If '{customer_id_col_fom8}' is 'Order ID', this cohort analysis is per order, not unique customer.")
                            else:
                                st.info("Not enough data to generate a cohort retention matrix.")

                    except Exception as e:
                        st.error(f"Error during Repeat Purchase Analysis: {e}")
        #st.markdown("---") # This markdown is now correctly AFTER the FOM expander

            # --- New Advanced Tool 1: ABC Analysis for Product Prioritization ---
        with st.expander("🥇 ABC Analysis for Product Prioritization", expanded=False):
                st.info("Classify products (SKU) into A, B, C categories based on their contribution to total revenue. 'A' items are high-value, 'C' are low-value.")
                all_cols_adv1 = df.columns.tolist()
                numeric_cols_adv1 = get_numeric_columns(df)

                sku_col_adv1 = st.selectbox("Select Product ID/SKU column:", all_cols_adv1, index=all_cols_adv1.index('SKU') if 'SKU' in all_cols_adv1 else 0, key="adv1_sku")
                amount_col_adv1 = st.selectbox("Select Sales Amount column:", numeric_cols_adv1, index=numeric_cols_adv1.index('Amount') if 'Amount' in numeric_cols_adv1 else 0, key="adv1_amount")

                if st.button("📊 Run ABC Analysis", key="adv1_run"):
                    if not sku_col_adv1 or not amount_col_adv1:
                        st.warning("Please select both SKU and Amount columns for ABC Analysis.")
                    else:
                        try:
                            abc_df = df[[sku_col_adv1, amount_col_adv1]].copy().dropna()
                            if abc_df.empty:
                                st.warning("No data available for ABC Analysis after filtering.")
                            else:
                                product_revenue = abc_df.groupby(sku_col_adv1)[amount_col_adv1].sum().sort_values(ascending=False).reset_index()
                                product_revenue['RevenueShare'] = product_revenue[amount_col_adv1] / product_revenue[amount_col_adv1].sum()
                                product_revenue['CumulativeRevenueShare'] = product_revenue['RevenueShare'].cumsum()

                                def abc_classification(cum_share):
                                    if cum_share <= 0.8:
                                        return 'A'
                                    elif cum_share <= 0.95:
                                        return 'B'
                                    else:
                                        return 'C'
                                product_revenue['ABC_Category'] = product_revenue['CumulativeRevenueShare'].apply(abc_classification)

                                st.markdown("##### ABC Analysis Results")
                                st.dataframe(product_revenue.head(20))

                                abc_summary = product_revenue.groupby('ABC_Category')[amount_col_adv1].agg(['sum', 'count']).rename(columns={'sum':'TotalRevenue', 'count':'NumberOfSKUs'})
                                abc_summary['RevenuePercentage'] = (abc_summary['TotalRevenue'] / abc_summary['TotalRevenue'].sum()) * 100
                                st.markdown("###### Summary by ABC Category")
                                st.dataframe(abc_summary)

                                fig_abc, ax_abc = plt.subplots()
                                abc_summary['TotalRevenue'].plot(kind='pie', autopct='%1.1f%%', ax=ax_abc, title='Revenue Contribution by ABC Category')
                                st.pyplot(fig_abc)

                        except Exception as e:
                            st.error(f"Error during ABC Analysis: {e}")
        

            # --- New Advanced Tool 2: Cross-Category Purchase Analysis ---
        with st.expander("🔗 Cross-Category Purchase Analysis", expanded=False):
                st.info("Identify which product categories are frequently purchased together within the same order. Requires 'Order ID' and 'Category' columns.")
                all_cols_adv2 = df.columns.tolist()
                
                order_id_col_adv2 = st.selectbox("Select Order ID column:", all_cols_adv2, index=all_cols_adv2.index('Order ID') if 'Order ID' in all_cols_adv2 else 0, key="adv2_order_id")
                category_col_adv2 = st.selectbox("Select Category column:", all_cols_adv2, index=all_cols_adv2.index('Category') if 'Category' in all_cols_adv2 else 0, key="adv2_category")

                if st.button("🤝 Run Cross-Category Analysis", key="adv2_run"):
                    if not order_id_col_adv2 or not category_col_adv2:
                        st.warning("Please select both Order ID and Category columns.")
                    else:
                        try:
                            cross_cat_df = df[[order_id_col_adv2, category_col_adv2]].copy().dropna().drop_duplicates()
                            if cross_cat_df.empty or cross_cat_df[category_col_adv2].nunique() < 2:
                                st.warning("Not enough data or distinct categories for cross-purchase analysis.")
                            else:
                                # Create a list of categories per order
                                order_categories = cross_cat_df.groupby(order_id_col_adv2)[category_col_adv2].apply(list).reset_index()
                                
                                # Create a co-occurrence matrix
                                from itertools import combinations
                                from collections import Counter

                                co_occurrence = Counter()
                                for categories_in_order in order_categories[category_col_adv2]:
                                    if len(categories_in_order) >= 2: # Need at least two categories to form a pair
                                        for cat_pair in combinations(sorted(list(set(categories_in_order))), 2):
                                            co_occurrence[cat_pair] += 1
                                
                                if not co_occurrence:
                                    st.info("No orders found with multiple distinct categories.")
                                else:
                                    co_occurrence_df = pd.DataFrame(co_occurrence.items(), columns=['CategoryPair', 'Frequency']).sort_values(by='Frequency', ascending=False)
                                    
                                    st.markdown("##### Top Co-purchased Category Pairs")
                                    st.dataframe(co_occurrence_df.head(15))

                                    # For heatmap (optional, can be large)
                                    if co_occurrence_df.shape[0] < 200 and cross_cat_df[category_col_adv2].nunique() < 25: # Limit heatmap size
                                        cat_list = sorted(cross_cat_df[category_col_adv2].unique())
                                        adj_matrix = pd.DataFrame(0, index=cat_list, columns=cat_list)
                                        for pair, freq in co_occurrence.items():
                                            adj_matrix.loc[pair[0], pair[1]] = freq
                                            adj_matrix.loc[pair[1], pair[0]] = freq # Symmetric
                                        
                                        fig_co, ax_co = plt.subplots(figsize=(max(8, len(cat_list)*0.5), max(6, len(cat_list)*0.4)))
                                        sns.heatmap(adj_matrix, annot=False, cmap="viridis", ax=ax_co)
                                        ax_co.set_title("Category Co-occurrence Heatmap")
                                        plt.tight_layout()
                                        st.pyplot(fig_co)
                                    else:
                                        st.info("Heatmap skipped due to large number of categories/pairs for better readability.")
                        except Exception as e:
                            st.error(f"Error during Cross-Category Purchase Analysis: {e}")
        

            # --- New Advanced Tool 3: Sales Volatility Analysis ---
        with st.expander("📉 Sales Volatility Analysis", expanded=False):
                st.info("Identify products with highly fluctuating sales (high coefficient of variation). Requires SKU, Date, and a sales value (Qty or Amount).")
                all_cols_adv3 = df.columns.tolist()
                numeric_cols_adv3 = get_numeric_columns(df)
                date_cols_adv3 = date_cols

                sku_col_adv3 = st.selectbox("Select Product ID/SKU column:", all_cols_adv3, index=all_cols_adv3.index('SKU') if 'SKU' in all_cols_adv3 else 0, key="adv3_sku")
                date_col_adv3 = st.selectbox("Select Date column:", date_cols_adv3, index=date_cols_adv3.index('Date') if 'Date' in date_cols_adv3 else 0, key="adv3_date")
                value_col_adv3 = st.selectbox("Select Sales Value column (Qty or Amount):", numeric_cols_adv3, index=numeric_cols_adv3.index('Qty') if 'Qty' in numeric_cols_adv3 else (numeric_cols_adv3.index('Amount') if 'Amount' in numeric_cols_adv3 else 0), key="adv3_value")
                agg_freq_adv3 = st.selectbox("Aggregation period for volatility:", ["W", "M"], index=1, format_func=lambda x: {"W":"Weekly", "M":"Monthly"}[x], key="adv3_freq")

                if st.button("🌪️ Analyze Sales Volatility", key="adv3_run"):
                    if not all([sku_col_adv3, date_col_adv3, value_col_adv3]):
                        st.warning("Please select SKU, Date, and Sales Value columns.")
                    else:
                        try:
                            vol_df = df[[sku_col_adv3, date_col_adv3, value_col_adv3]].copy()
                            vol_df[date_col_adv3] = pd.to_datetime(vol_df[date_col_adv3], errors='coerce')
                            vol_df = vol_df.dropna()

                            if vol_df.empty:
                                st.warning("No data for volatility analysis after filtering.")
                            else:
                                sales_per_period = vol_df.groupby([sku_col_adv3, pd.Grouper(key=date_col_adv3, freq=agg_freq_adv3)])[value_col_adv3].sum().reset_index()
                                
                                # Calculate Coefficient of Variation (std_dev / mean)
                                volatility_stats = sales_per_period.groupby(sku_col_adv3)[value_col_adv3].agg(['mean', 'std']).reset_index()
                                volatility_stats['CoV'] = (volatility_stats['std'] / volatility_stats['mean']).fillna(0) # Handle mean=0
                                volatility_stats = volatility_stats.sort_values(by='CoV', ascending=False)

                                st.markdown("##### Sales Volatility (Coefficient of Variation)")
                                st.dataframe(volatility_stats.head(20))

                                fig_vol, ax_vol = plt.subplots()
                                sns.histplot(volatility_stats[volatility_stats['CoV'] > 0]['CoV'], kde=True, ax=ax_vol, bins=30)
                                ax_vol.set_title(f"Distribution of Sales Volatility (CoV) - {agg_freq_adv3} aggregation")
                                st.pyplot(fig_vol)
                        except Exception as e:
                            st.error(f"Error during Sales Volatility Analysis: {e}")
        #st.markdown("---")

            # --- New Advanced Tool 4: Customer Acquisition Cohort Value Analysis ---
        with st.expander("📈 Customer Acquisition Cohort Value Analysis", expanded=False):
                st.info("Track the average spending of new customer cohorts (based on first purchase month) over subsequent months.")
                all_cols_adv4 = df.columns.tolist()
                numeric_cols_adv4 = get_numeric_columns(df)
                date_cols_adv4 = date_cols

                cust_id_col_adv4 = st.selectbox("Select Customer ID column (or proxy):", all_cols_adv4, index=all_cols_adv4.index('Order ID') if 'Order ID' in all_cols_adv4 else 0, key="adv4_cust_id")
                date_col_adv4 = st.selectbox("Select Order Date column:", date_cols_adv4, index=date_cols_adv4.index('Date') if 'Date' in date_cols_adv4 else 0, key="adv4_date")
                amount_col_adv4 = st.selectbox("Select Sales Amount column:", numeric_cols_adv4, index=numeric_cols_adv4.index('Amount') if 'Amount' in numeric_cols_adv4 else 0, key="adv4_amount")

                if st.button("👥 Analyze Acquisition Cohort Value", key="adv4_run"):
                    if not all([cust_id_col_adv4, date_col_adv4, amount_col_adv4]):
                        st.warning("Please select Customer ID, Date, and Amount columns.")
                    else:
                        try:
                            cohort_val_df = df[[cust_id_col_adv4, date_col_adv4, amount_col_adv4]].copy()
                            cohort_val_df[date_col_adv4] = pd.to_datetime(cohort_val_df[date_col_adv4], errors='coerce')
                            cohort_val_df = cohort_val_df.dropna()

                            if cohort_val_df.empty:
                                st.warning("No data for cohort value analysis.")
                            else:
                                cohort_val_df['OrderMonth'] = cohort_val_df[date_col_adv4].dt.to_period('M')
                                cohort_val_df['CohortMonth'] = cohort_val_df.groupby(cust_id_col_adv4)['OrderMonth'].transform('min')
                                
                                def get_cohort_period_adv(df_cohort, event_month_col='OrderMonth', cohort_month_col='CohortMonth'):
                                    df_cohort['CohortPeriod'] = (df_cohort[event_month_col].dt.year - df_cohort[cohort_month_col].dt.year) * 12 + \
                                                              (df_cohort[event_month_col].dt.month - df_cohort[cohort_month_col].dt.month)
                                    return df_cohort
                                cohort_val_df = get_cohort_period_adv(cohort_val_df)

                                cohort_data = cohort_val_df.groupby(['CohortMonth', 'CohortPeriod']).agg(
                                    TotalSpend=(amount_col_adv4, 'sum'),
                                    UniqueCustomers=(cust_id_col_adv4, 'nunique')
                                ).reset_index()
                                cohort_data['AvgSpendPerCustomer'] = cohort_data['TotalSpend'] / cohort_data['UniqueCustomers']
                                
                                cohort_pivot_value = cohort_data.pivot_table(index='CohortMonth', columns='CohortPeriod', values='AvgSpendPerCustomer')
                                
                                st.markdown("##### Average Spend per Customer by Acquisition Cohort (Monthly)")
                                if not cohort_pivot_value.empty:
                                    fig_cohort_val, ax_cohort_val = plt.subplots(figsize=(12, max(6, len(cohort_pivot_value)*0.4)))
                                    sns.heatmap(cohort_pivot_value, annot=True, fmt='.0f', cmap="Greens", ax=ax_cohort_val) # fmt for currency
                                    ax_cohort_val.set_title('Avg. Monthly Spend per Customer by Acquisition Cohort')
                                    ax_cohort_val.set_xlabel('Months Since Acquisition')
                                    ax_cohort_val.set_ylabel('Acquisition Cohort (First Purchase Month)')
                                    plt.tight_layout()
                                    st.pyplot(fig_cohort_val)
                                    st.caption(f"Note: If '{cust_id_col_adv4}' is 'Order ID', this analysis is per order, not unique customer.")
                                else:
                                    st.info("Not enough data to generate cohort value matrix.")
                        except Exception as e:
                            st.error(f"Error during Customer Acquisition Cohort Value Analysis: {e}")
        #st.markdown("---")

            # --- New Advanced Tool 5: High-Value vs. Low-Value Order Profiling ---
        with st.expander("💰 High-Value vs. Low-Value Order Profiling", expanded=False):
                st.info("Compare characteristics of high-value orders vs. low-value orders based on a defined threshold.")
                all_cols_adv5 = df.columns.tolist()
                numeric_cols_adv5 = get_numeric_columns(df)
                categorical_cols_adv5 = get_categorical_columns(df)

                order_id_col_adv5 = st.selectbox("Select Order ID column:", all_cols_adv5, index=all_cols_adv5.index('Order ID') if 'Order ID' in all_cols_adv5 else 0, key="adv5_order_id")
                amount_col_adv5 = st.selectbox("Select Order Amount column:", numeric_cols_adv5, index=numeric_cols_adv5.index('Amount') if 'Amount' in numeric_cols_adv5 else 0, key="adv5_amount")
                profiling_cols_adv5 = st.multiselect("Select categorical columns for profiling (e.g., Category, Sales Channel):", categorical_cols_adv5, default=[col for col in ['Category', 'Sales Channel', 'Fulfilment', 'B2B'] if col in categorical_cols_adv5], key="adv5_profile_cols")
                
                # Assuming each row is an item, first aggregate by Order ID if 'Amount' is item amount
                # For this dataset, 'Order ID' is unique per row, so 'Amount' is effectively order amount.
                # If 'Order ID' could group items, we'd sum 'Amount' first.
                value_threshold_adv5 = st.number_input("Define threshold for 'High-Value' order (e.g., top 25% quantile):", value=df[amount_col_adv5].quantile(0.75) if amount_col_adv5 in df.columns and not df[amount_col_adv5].empty else 100.0, key="adv5_threshold")

                if st.button("🔍 Profile Order Values", key="adv5_run"):
                    if not order_id_col_adv5 or not amount_col_adv5:
                        st.warning("Please select Order ID and Amount columns.")
                    else:
                        try:
                            profile_df = df.copy() # Use all columns for profiling
                            profile_df['OrderValueSegment'] = np.where(profile_df[amount_col_adv5] >= value_threshold_adv5, 'High-Value', 'Low-Value')
                            
                            st.markdown(f"##### Profiling High-Value (>= {value_threshold_adv5}) vs. Low-Value Orders")
                            st.write("Segment Sizes:")
                            st.write(profile_df['OrderValueSegment'].value_counts())

                            for col_to_profile in profiling_cols_adv5:
                                if col_to_profile in profile_df.columns:
                                    st.markdown(f"###### Distribution of '{col_to_profile}' by Order Value Segment")
                                    summary_table = pd.crosstab(profile_df[col_to_profile], profile_df['OrderValueSegment'], normalize='columns') * 100
                                    st.dataframe(summary_table.round(1))
                                    
                                    # Avoid plotting if too many unique values in the profiling column
                                    if profile_df[col_to_profile].nunique() < 15:
                                        fig_profile, ax_profile = plt.subplots()
                                        sns.countplot(data=profile_df, x=col_to_profile, hue='OrderValueSegment', ax=ax_profile)
                                        ax_profile.set_title(f"'{col_to_profile}' by Order Value Segment")
                                        plt.xticks(rotation=45, ha="right")
                                        plt.tight_layout()
                                        st.pyplot(fig_profile)
                                    else:
                                        st.info(f"Plot for '{col_to_profile}' skipped due to high cardinality ({profile_df[col_to_profile].nunique()} unique values). Table provided above.")
                        except Exception as e:
                            st.error(f"Error during Order Value Profiling: {e}")
        #st.markdown("---")

            # --- New Advanced Tool 6: Product Return Rate by Attribute ---
        with st.expander("↩️ Product Return Rate by Attribute", expanded=False):
                st.info("Analyze product return rates grouped by a selected attribute (e.g., Category, Style, Size). Requires a Product ID, Return Indicator, and an Attribute column.")
                all_cols_adv6 = df.columns.tolist()
                categorical_cols_adv6 = get_categorical_columns(df, nunique_threshold=50) # Allow more unique values for attributes

                product_id_col_adv6 = st.selectbox("Select Product ID column (e.g., SKU):", all_cols_adv6, index=all_cols_adv6.index('SKU') if 'SKU' in all_cols_adv6 else 0, key="adv6_pid")
                return_indicator_col_adv6 = st.selectbox("Select Return Indicator column (binary 0/1 or True/False):", [None] + all_cols_adv6, index=0, key="adv6_return_indicator")
                attribute_col_adv6 = st.selectbox("Select Attribute column for grouping (e.g., Category, Style):", [None] + categorical_cols_adv6, index=0, key="adv6_attribute")

                if st.button("📉 Analyze Return Rate by Attribute", key="adv6_run"):
                    if not all([product_id_col_adv6, return_indicator_col_adv6, attribute_col_adv6]):
                        st.warning("Please select Product ID, Return Indicator, and Attribute columns.")
                    elif return_indicator_col_adv6 not in df.columns:
                        st.warning(f"Return Indicator column '{return_indicator_col_adv6}' not found.")
                    elif attribute_col_adv6 not in df.columns:
                        st.warning(f"Attribute column '{attribute_col_adv6}' not found.")
                    else:
                        try:
                            return_attr_df = df[[product_id_col_adv6, return_indicator_col_adv6, attribute_col_adv6]].copy().dropna()
                            
                            # Process return indicator to be binary 0/1
                            if return_attr_df[return_indicator_col_adv6].dtype == 'object':
                                true_vals_adv6 = ['true', 'yes', '1', 'returned', 'shipped - returned to seller'] 
                                false_vals_adv6 = ['false', 'no', '0', 'not returned', 'shipped']
                                return_attr_df['IsReturnedBinary'] = return_attr_df[return_indicator_col_adv6].astype(str).str.lower().map(lambda x: 1 if x in true_vals_adv6 else (0 if x in false_vals_adv6 else pd.NA))
                            else:
                                return_attr_df['IsReturnedBinary'] = pd.to_numeric(return_attr_df[return_indicator_col_adv6], errors='coerce')
                            
                            return_attr_df = return_attr_df.dropna(subset=['IsReturnedBinary'])

                            if not return_attr_df['IsReturnedBinary'].isin([0,1]).all():
                                st.warning(f"Column '{return_indicator_col_adv6}' must be binary (0/1) after processing. Check its values.")
                            elif return_attr_df.empty:
                                st.warning("No valid data for return analysis by attribute.")
                            else:
                                return_rate_by_attr = return_attr_df.groupby(attribute_col_adv6)['IsReturnedBinary'].agg(['mean', 'count'])
                                return_rate_by_attr.columns = ['ReturnRate', 'TotalItems']
                                return_rate_by_attr['ReturnRate'] = return_rate_by_attr['ReturnRate'] * 100 # As percentage
                                return_rate_by_attr = return_rate_by_attr.sort_values(by='ReturnRate', ascending=False)

                                st.markdown(f"##### Return Rate by '{attribute_col_adv6}'")
                                st.dataframe(return_rate_by_attr.head(20))

                                if not return_rate_by_attr.empty and return_rate_by_attr.shape[0] < 30 : # Limit plot for readability
                                    fig_ret_attr, ax_ret_attr = plt.subplots(figsize=(10, max(5, return_rate_by_attr.shape[0]*0.3)))
                                    return_rate_by_attr['ReturnRate'].sort_values(ascending=True).plot(kind='barh', ax=ax_ret_attr)
                                    ax_ret_attr.set_title(f"Return Rate (%) by {attribute_col_adv6}")
                                    ax_ret_attr.set_xlabel("Return Rate (%)")
                                    plt.tight_layout()
                                    st.pyplot(fig_ret_attr)
                                else:
                                    st.info("Bar chart skipped due to high number of attribute values. Table provided above.")
                        except Exception as e:
                            st.error(f"Error during Return Rate by Attribute analysis: {e}")
        #st.markdown("---")

            # --- New Advanced Tool 7: Fulfillment Efficiency by Region ---
        with st.expander("🚚 Fulfillment Efficiency by Region", expanded=False):
                st.info("Compare fulfillment methods' performance (e.g., AOV, quantity per order) across different regions.")
                all_cols_adv7 = df.columns.tolist()
                numeric_cols_adv7 = get_numeric_columns(df)
                categorical_cols_adv7 = get_categorical_columns(df)

                fulfilment_col_adv7 = st.selectbox("Select Fulfilment column:", all_cols_adv7, index=all_cols_adv7.index('Fulfilment') if 'Fulfilment' in all_cols_adv7 else 0, key="adv7_fulfilment")
                region_col_adv7 = st.selectbox("Select Region column (e.g., ship-state):", all_cols_adv7, index=all_cols_adv7.index('ship-state') if 'ship-state' in all_cols_adv7 else 0, key="adv7_region")
                amount_col_adv7 = st.selectbox("Select Sales Amount column:", numeric_cols_adv7, index=numeric_cols_adv7.index('Amount') if 'Amount' in numeric_cols_adv7 else 0, key="adv7_amount")
                qty_col_adv7 = st.selectbox("Select Quantity Sold column:", numeric_cols_adv7, index=numeric_cols_adv7.index('Qty') if 'Qty' in numeric_cols_adv7 else 0, key="adv7_qty")
                order_id_col_adv7 = st.selectbox("Select Order ID column:", all_cols_adv7, index=all_cols_adv7.index('Order ID') if 'Order ID' in all_cols_adv7 else 0, key="adv7_order_id")

                if st.button("🗺️ Analyze Fulfillment by Region", key="adv7_run"):
                    if not all([fulfilment_col_adv7, region_col_adv7, amount_col_adv7, qty_col_adv7, order_id_col_adv7]):
                        st.warning("Please select all required columns.")
                    else:
                        try:
                            ff_region_df = df[[fulfilment_col_adv7, region_col_adv7, amount_col_adv7, qty_col_adv7, order_id_col_adv7]].copy().dropna()
                            if ff_region_df.empty:
                                st.warning("No data for fulfillment by region analysis.")
                            else:
                                # Aggregate by Order ID first if necessary, then by fulfilment/region
                                # Assuming 'Order ID' is unique per row for this dataset's structure for Amount/Qty
                                regional_ff_summary = ff_region_df.groupby([region_col_adv7, fulfilment_col_adv7]).agg(
                                    TotalRevenue=(amount_col_adv7, 'sum'),
                                    TotalQuantity=(qty_col_adv7, 'sum'),
                                    NumberOfOrders=(order_id_col_adv7, 'nunique') # Number of unique orders
                                ).reset_index()
                                regional_ff_summary['AOV'] = regional_ff_summary['TotalRevenue'] / regional_ff_summary['NumberOfOrders'].replace(0, np.nan)
                                regional_ff_summary['AvgQtyPerOrder'] = regional_ff_summary['TotalQuantity'] / regional_ff_summary['NumberOfOrders'].replace(0, np.nan)
                                regional_ff_summary = regional_ff_summary.dropna(subset=['AOV'])

                                st.markdown(f"##### Fulfillment Performance by '{region_col_adv7}' and '{fulfilment_col_adv7}'")
                                st.dataframe(regional_ff_summary.sort_values(by=[region_col_adv7, 'TotalRevenue'], ascending=[True, False]).head(30))

                                # Plot AOV by fulfilment method for top N regions
                                top_n_regions_adv7 = 5
                                top_regions = ff_region_df.groupby(region_col_adv7)[amount_col_adv7].sum().nlargest(top_n_regions_adv7).index
                                
                                if not top_regions.empty:
                                    plot_data_adv7 = regional_ff_summary[regional_ff_summary[region_col_adv7].isin(top_regions)]
                                    if not plot_data_adv7.empty:
                                        fig_ff_region, ax_ff_region = plt.subplots(figsize=(12,7))
                                        sns.barplot(data=plot_data_adv7, x=region_col_adv7, y='AOV', hue=fulfilment_col_adv7, ax=ax_ff_region, order=top_regions)
                                        ax_ff_region.set_title(f"Average Order Value (AOV) by Fulfillment & Region (Top {top_n_regions_adv7} Regions)")
                                        plt.xticks(rotation=45, ha="right")
                                        plt.tight_layout()
                                        st.pyplot(fig_ff_region)
                                    else:
                                        st.info("Not enough data for plotting AOV for top regions.")
                                else:
                                    st.info("Not enough regions to plot.")
                        except Exception as e:
                            st.error(f"Error during Fulfillment Efficiency by Region analysis: {e}")
        #st.markdown("---")

            # --- New Advanced Tool 8: B2B vs. B2C Sales Dynamics ---
        with st.expander("🏢 B2B vs. B2C Sales Dynamics", expanded=False):
                st.info("Deeper dive into B2B vs. B2C sales patterns (e.g., product preferences, AOV over time). Requires a B2B indicator column.")
                all_cols_adv8 = df.columns.tolist()
                numeric_cols_adv8 = get_numeric_columns(df)
                date_cols_adv8 = date_cols
                categorical_cols_adv8 = get_categorical_columns(df)

                b2b_col_adv8 = st.selectbox("Select B2B indicator column (boolean/binary):", [None] + all_cols_adv8, index=([None] + all_cols_adv8).index('B2B') if 'B2B' in all_cols_adv8 else 0, key="adv8_b2b")
                date_col_adv8 = st.selectbox("Select Date column:", date_cols_adv8, index=date_cols_adv8.index('Date') if 'Date' in date_cols_adv8 else 0, key="adv8_date")
                amount_col_adv8 = st.selectbox("Select Sales Amount column:", numeric_cols_adv8, index=numeric_cols_adv8.index('Amount') if 'Amount' in numeric_cols_adv8 else 0, key="adv8_amount")
                qty_col_adv8 = st.selectbox("Select Quantity Sold column:", numeric_cols_adv8, index=numeric_cols_adv8.index('Qty') if 'Qty' in numeric_cols_adv8 else 0, key="adv8_qty")
                order_id_col_adv8 = st.selectbox("Select Order ID column:", all_cols_adv8, index=all_cols_adv8.index('Order ID') if 'Order ID' in all_cols_adv8 else 0, key="adv8_order_id")
                product_level_col_adv8 = st.selectbox("Analyze product preference by (e.g., Category, SKU):", [None] + categorical_cols_adv8, index=([None] + categorical_cols_adv8).index('Category') if 'Category' in categorical_cols_adv8 else 0, key="adv8_product_level")
                agg_freq_adv8 = st.selectbox("AOV trend aggregation:", ["M", "Q"], index=0, format_func=lambda x: {"M":"Monthly", "Q":"Quarterly"}[x], key="adv8_freq")

                if st.button("💼 Analyze B2B vs. B2C Dynamics", key="adv8_run"):
                    if not b2b_col_adv8 or b2b_col_adv8 not in df.columns:
                        st.warning("Please select a valid B2B indicator column.")
                    elif not all([date_col_adv8, amount_col_adv8, qty_col_adv8, order_id_col_adv8]):
                        st.warning("Please select Date, Amount, Quantity, and Order ID columns.")
                    else:
                        try:
                            b2b_df = df.copy()
                            # Ensure B2B column is boolean-like
                            if b2b_df[b2b_col_adv8].dtype == 'object':
                                true_vals_adv8 = ['true', 'yes', '1', 'b2b']
                                b2b_df['IsB2B_Processed'] = b2b_df[b2b_col_adv8].astype(str).str.lower().isin(true_vals_adv8)
                            else:
                                b2b_df['IsB2B_Processed'] = b2b_df[b2b_col_adv8].astype(bool)
                            
                            b2b_df[date_col_adv8] = pd.to_datetime(b2b_df[date_col_adv8], errors='coerce')
                            b2b_df = b2b_df.dropna(subset=[date_col_adv8, amount_col_adv8, qty_col_adv8, order_id_col_adv8, 'IsB2B_Processed'])

                            if b2b_df.empty:
                                st.warning("No data for B2B/B2C analysis after filtering.")
                            else:
                                st.markdown("##### B2B vs. B2C Sales Dynamics")
                                
                                # AOV Trend
                                b2b_df['TimePeriod'] = b2b_df[date_col_adv8].dt.to_period(agg_freq_adv8)
                                aov_trend_b2b = b2b_df.groupby(['TimePeriod', 'IsB2B_Processed']).agg(
                                    TotalRevenue=(amount_col_adv8, 'sum'),
                                    UniqueOrders=(order_id_col_adv8, 'nunique')
                                )
                                aov_trend_b2b['AOV'] = aov_trend_b2b['TotalRevenue'] / aov_trend_b2b['UniqueOrders'].replace(0, np.nan)
                                aov_plot_data = aov_trend_b2b['AOV'].unstack().rename(columns={True: 'B2B', False: 'B2C'})
                                
                                if not aov_plot_data.empty:
                                    st.markdown(f"###### AOV Trend ({agg_freq_adv8}) for B2B vs. B2C")
                                    st.line_chart(aov_plot_data.fillna(0))
                                else:
                                    st.info("Not enough data for AOV trend by B2B/B2C.")

                                # Product Preferences
                                if product_level_col_adv8 and product_level_col_adv8 in b2b_df.columns:
                                    st.markdown(f"###### Top 5 {product_level_col_adv8} by Revenue for B2B vs. B2C")
                                    top_products_b2b = b2b_df[b2b_df['IsB2B_Processed']].groupby(product_level_col_adv8)[amount_col_adv8].sum().nlargest(5)
                                    top_products_b2c = b2b_df[~b2b_df['IsB2B_Processed']].groupby(product_level_col_adv8)[amount_col_adv8].sum().nlargest(5)
                                    
                                    col_b2b, col_b2c = st.columns(2)
                                    with col_b2b:
                                        st.write("B2B Top Products:")
                                        st.dataframe(top_products_b2b)
                                    with col_b2c:
                                        st.write("B2C (Non-B2B) Top Products:")
                                        st.dataframe(top_products_b2c)
                                else:
                                    st.info("Product level column not selected for preference analysis.")
                        except Exception as e:
                            st.error(f"Error during B2B vs. B2C Sales Dynamics analysis: {e}")
        #st.markdown("---")
        # --- New Advanced Tool 9: Sales Seasonality Index by Category ---
        with st.expander("📅 Sales Seasonality Index by Category", expanded=False): # This is still part of tab3
                st.info("Calculate and visualize a seasonality index for different product categories to understand periodic sales fluctuations.")
                all_cols_adv9 = df.columns.tolist()
                numeric_cols_adv9 = get_numeric_columns(df)
                date_cols_adv9 = date_cols
                categorical_cols_adv9 = get_categorical_columns(df)

                date_col_adv9 = st.selectbox("Select Date column:", date_cols_adv9, index=date_cols_adv9.index('Date') if 'Date' in date_cols_adv9 else 0, key="adv9_date")
                amount_col_adv9 = st.selectbox("Select Sales Amount column:", numeric_cols_adv9, index=numeric_cols_adv9.index('Amount') if 'Amount' in numeric_cols_adv9 else 0, key="adv9_amount")
                category_col_adv9 = st.selectbox("Select Category column:", categorical_cols_adv9, index=categorical_cols_adv9.index('Category') if 'Category' in categorical_cols_adv9 else 0, key="adv9_category")
                
                if st.button("📈 Analyze Category Seasonality", key="adv9_run"):
                    if not all([date_col_adv9, amount_col_adv9, category_col_adv9]):
                        st.warning("Please select Date, Amount, and Category columns.")
                    else:
                        try:
                            season_df = df[[date_col_adv9, amount_col_adv9, category_col_adv9]].copy()
                            season_df[date_col_adv9] = pd.to_datetime(season_df[date_col_adv9], errors='coerce')
                            season_df = season_df.dropna()

                            if season_df.empty:
                                st.warning("No data for seasonality analysis.")
                            else:
                                # Resample to monthly sales by category
                                monthly_sales_cat = season_df.groupby([category_col_adv9, pd.Grouper(key=date_col_adv9, freq='M')])[amount_col_adv9].sum().reset_index()
                                monthly_sales_cat['MonthOfYear'] = monthly_sales_cat[date_col_adv9].dt.month
                                
                                if monthly_sales_cat.empty:
                                    st.warning("Not enough data after monthly aggregation.")
                                else:
                                    # Calculate average monthly sales for each category
                                    avg_monthly_sales_overall = monthly_sales_cat.groupby(category_col_adv9)[amount_col_adv9].mean().reset_index()
                                    avg_monthly_sales_overall = avg_monthly_sales_overall.rename(columns={amount_col_adv9: 'AvgOverallMonthlySales'})

                                    # Calculate average sales for each specific month of the year, per category
                                    avg_sales_by_month_cat = monthly_sales_cat.groupby([category_col_adv9, 'MonthOfYear'])[amount_col_adv9].mean().reset_index()
                                    avg_sales_by_month_cat = avg_sales_by_month_cat.rename(columns={amount_col_adv9: 'AvgSalesForThisMonth'})

                                    # Merge to calculate seasonality index
                                    seasonality_index_df = pd.merge(avg_sales_by_month_cat, avg_monthly_sales_overall, on=category_col_adv9)
                                    seasonality_index_df['SeasonalityIndex'] = (seasonality_index_df['AvgSalesForThisMonth'] / seasonality_index_df['AvgOverallMonthlySales'].replace(0, np.nan)) * 100
                                    seasonality_index_df = seasonality_index_df.dropna(subset=['SeasonalityIndex'])
                                    
                                    st.markdown(f"##### Sales Seasonality Index by '{category_col_adv9}' (Index 100 = Average)")
                                    
                                    # Pivot for plotting
                                    pivot_seasonality = seasonality_index_df.pivot_table(index='MonthOfYear', columns=category_col_adv9, values='SeasonalityIndex')
                                    
                                    if not pivot_seasonality.empty:
                                        # Select top N categories by total sales for cleaner plot
                                        top_n_cats_plot = 5
                                        top_cats = season_df.groupby(category_col_adv9)[amount_col_adv9].sum().nlargest(top_n_cats_plot).index
                                        
                                        if not top_cats.empty and all(cat in pivot_seasonality.columns for cat in top_cats):
                                            st.line_chart(pivot_seasonality[top_cats].fillna(100)) # Fill NA with 100 (avg) for plot
                                            st.dataframe(pivot_seasonality[top_cats].head(12))
                                        elif not pivot_seasonality.columns.empty: # Fallback if top_cats logic fails
                                            st.line_chart(pivot_seasonality.iloc[:, :min(top_n_cats_plot, len(pivot_seasonality.columns))].fillna(100))
                                            st.dataframe(pivot_seasonality.head(12))
                                        else:
                                            st.info("Not enough category data to plot seasonality index.")
                                    else:
                                        st.info("Could not compute seasonality index pivot table.")
                        except Exception as e:
                            st.error(f"Error during Sales Seasonality Index analysis: {e}")
        #st.markdown("---")
        # --- New Advanced Tool 10: Promotion Effectiveness by Product Category ---
        with st.expander("🎉 Promotion Effectiveness by Product Category", expanded=False):
                st.info("Analyze if promotions are more effective for certain product categories by comparing AOV or Quantity lift for promoted vs. non-promoted items within those categories.")
                all_cols_adv10 = df.columns.tolist()
                numeric_cols_adv10 = get_numeric_columns(df)
                categorical_cols_adv10 = get_categorical_columns(df)

                promo_ids_col_adv10 = st.selectbox("Select Promotion IDs column:", all_cols_adv10, index=all_cols_adv10.index('promotion-ids') if 'promotion-ids' in all_cols_adv10 else 0, key="adv10_promo_ids")
                amount_col_adv10 = st.selectbox("Select Sales Amount column:", numeric_cols_adv10, index=numeric_cols_adv10.index('Amount') if 'Amount' in numeric_cols_adv10 else 0, key="adv10_amount")
                qty_col_adv10 = st.selectbox("Select Quantity Sold column:", numeric_cols_adv10, index=numeric_cols_adv10.index('Qty') if 'Qty' in numeric_cols_adv10 else 0, key="adv10_qty")
                category_col_adv10 = st.selectbox("Select Category column:", categorical_cols_adv10, index=categorical_cols_adv10.index('Category') if 'Category' in categorical_cols_adv10 else 0, key="adv10_category")
                order_id_col_adv10 = st.selectbox("Select Order ID column:", all_cols_adv10, index=all_cols_adv10.index('Order ID') if 'Order ID' in all_cols_adv10 else 0, key="adv10_order_id") # For AOV

                if st.button("🚀 Analyze Promotion Effectiveness by Category", key="adv10_run"):
                    if not all([promo_ids_col_adv10, amount_col_adv10, qty_col_adv10, category_col_adv10, order_id_col_adv10]):
                        st.warning("Please select all required columns.")
                    elif not df[promo_ids_col_adv10].isnull().all() == False and df[promo_ids_col_adv10].astype(str).str.strip().eq('').all() == False : # Check if promo col has data
                        st.warning(f"The promotion ID column '{promo_ids_col_adv10}' seems to be empty or all NaNs. Cannot analyze promotion effectiveness.")
                    else:
                        try:
                            promo_cat_df = df[[promo_ids_col_adv10, amount_col_adv10, qty_col_adv10, category_col_adv10, order_id_col_adv10]].copy()
                            promo_cat_df['HasPromotion'] = ~promo_cat_df[promo_ids_col_adv10].isnull() & (promo_cat_df[promo_ids_col_adv10].astype(str).str.strip() != '')
                            promo_cat_df = promo_cat_df.dropna(subset=[amount_col_adv10, qty_col_adv10, category_col_adv10, order_id_col_adv10])

                            if promo_cat_df.empty:
                                st.warning("No data for promotion effectiveness by category analysis.")
                            elif not promo_cat_df['HasPromotion'].any():
                                st.info("No items marked with promotions found in the selected data based on the Promotion IDs column.")
                            else:
                                # Calculate AOV and Avg Qty per Order, by Category and Promotion Status
                                # This is item-level, so AOV here would be average item price.
                                # For true AOV, would need to group by Order ID first.
                                # Let's calculate average item price and average quantity for simplicity here.
                                
                                summary_promo_cat = promo_cat_df.groupby([category_col_adv10, 'HasPromotion']).agg(
                                    AvgItemPrice=(amount_col_adv10, 'mean'),
                                    AvgItemQty=(qty_col_adv10, 'mean'),
                                    TotalRevenue=(amount_col_adv10, 'sum'),
                                    TotalItems=(order_id_col_adv10, 'count') # Number of line items
                                ).reset_index()
                                
                                # Pivot to compare Promoted vs. Non-Promoted side-by-side
                                pivot_avg_price = summary_promo_cat.pivot_table(index=category_col_adv10, columns='HasPromotion', values='AvgItemPrice')
                                if True in pivot_avg_price.columns and False in pivot_avg_price.columns: # Ensure both promo and non-promo exist
                                    pivot_avg_price['PriceLiftRatio'] = (pivot_avg_price[True] / pivot_avg_price[False].replace(0,np.nan))
                                    pivot_avg_price.columns = ['AvgPrice_NonPromo', 'AvgPrice_Promo', 'PriceLiftRatio_Promo']
                                
                                pivot_avg_qty = summary_promo_cat.pivot_table(index=category_col_adv10, columns='HasPromotion', values='AvgItemQty')
                                if True in pivot_avg_qty.columns and False in pivot_avg_qty.columns:
                                    pivot_avg_qty['QtyLiftRatio'] = (pivot_avg_qty[True] / pivot_avg_qty[False].replace(0,np.nan))
                                    pivot_avg_qty.columns = ['AvgQty_NonPromo', 'AvgQty_Promo', 'QtyLiftRatio_Promo']

                                st.markdown(f"##### Promotion Effectiveness by '{category_col_adv10}'")
                                if not pivot_avg_price.empty:
                                    st.markdown("###### Average Item Price Comparison (Promoted vs. Non-Promoted)")
                                    st.dataframe(pivot_avg_price.dropna(subset=['PriceLiftRatio_Promo']).sort_values(by='PriceLiftRatio_Promo', ascending=False).head(15))
                                else:
                                    st.info("Not enough data to compare average item prices by promotion status within categories.")

                                if not pivot_avg_qty.empty:
                                    st.markdown("###### Average Item Quantity Comparison (Promoted vs. Non-Promoted)")
                                    st.dataframe(pivot_avg_qty.dropna(subset=['QtyLiftRatio_Promo']).sort_values(by='QtyLiftRatio_Promo', ascending=False).head(15))
                                else:
                                    st.info("Not enough data to compare average item quantities by promotion status within categories.")

                                # Top categories by revenue from promotions
                                top_promo_revenue_cat = summary_promo_cat[summary_promo_cat['HasPromotion']].groupby(category_col_adv10)['TotalRevenue'].sum().nlargest(10)
                                if not top_promo_revenue_cat.empty:
                                    st.markdown("###### Top Categories by Revenue from Promoted Items")
                                    st.bar_chart(top_promo_revenue_cat)

                        except Exception as e:
                            st.error(f"Error during Promotion Effectiveness by Category analysis: {e}")
        #st.markdown("---")
        with st.expander("🎯 Transaction B2B Status Prediction"): # This is still part of tab3
            # st.subheader("") # Removed empty subheader as expander title is descriptive
            st.info("Train a model to predict if a transaction is B2B based on its features. Select your features and the target B2B column.")

            all_cols_mls1 = df.columns.tolist()
            numeric_cols_mls1 = get_numeric_columns(df)
            # For features, allow selection from numeric and lower-cardinality categorical
            categorical_features_mls1 = get_categorical_columns(df, nunique_threshold=50) 

            # Target column selection - should be a binary column like 'B2B'
            # Default to 'B2B' if it exists and seems appropriate (boolean or 2 unique values)
            default_target_mls1_index = 0
            if 'B2B' in all_cols_mls1 and (df['B2B'].dtype == 'bool' or df['B2B'].nunique() <= 2):
                default_target_mls1_index = all_cols_mls1.index('B2B')
            
            target_col_mls1 = st.selectbox("Select Target Column (Binary - e.g., 'B2B'):", 
                                           all_cols_mls1, 
                                           index=default_target_mls1_index, 
                                           key="mls1_target")

            available_features_mls1 = [col for col in numeric_cols_mls1 + categorical_features_mls1 if col != target_col_mls1]
            # Sensible defaults for features, excluding ID-like columns and the target
            default_features_mls1 = [f for f in ['Amount', 'Qty', 'Category', 'Sales Channel', 'Fulfilment', 'ship-state'] if f in available_features_mls1]
            
            selected_features_mls1 = st.multiselect("Select Feature Columns:", 
                                                    available_features_mls1, 
                                                    default=default_features_mls1, 
                                                    key="mls1_features")

            if st.button("🚀 Train B2B Prediction Model", key="mls1_run"):
                if not target_col_mls1 or not selected_features_mls1:
                    st.warning("Please select a target column and at least one feature column.")
                elif df[target_col_mls1].nunique() > 2 :
                    st.warning(f"Target column '{target_col_mls1}' must be binary (have 2 unique values). It has {df[target_col_mls1].nunique()}.")
                else:
                    try:
                        mls1_X = df[selected_features_mls1].copy()
                        # Process target: ensure it's 0/1
                        if df[target_col_mls1].dtype == 'bool':
                            mls1_y = df[target_col_mls1].astype(int)
                        else: # Attempt to map if it's not bool but has 2 unique values
                            unique_vals = df[target_col_mls1].dropna().unique()
                            if len(unique_vals) == 2:
                                st.info(f"Mapping target '{target_col_mls1}': {unique_vals[0]} -> 0, {unique_vals[1]} -> 1")
                                mls1_y = df[target_col_mls1].map({unique_vals[0]: 0, unique_vals[1]: 1})
                            else: # Should have been caught by nunique check, but as a safeguard
                                st.error(f"Target column '{target_col_mls1}' could not be converted to binary 0/1.")
                                st.stop()
                        
                        mls1_X = mls1_X.loc[mls1_y.dropna().index] # Align X with y after potential NA drop from y mapping
                        mls1_y = mls1_y.dropna()

                        # Preprocessing: Impute NaNs and One-Hot Encode
                        for col in mls1_X.select_dtypes(include=np.number).columns:
                            mls1_X[col] = mls1_X[col].fillna(mls1_X[col].median())
                        for col in mls1_X.select_dtypes(include='object').columns:
                            mls1_X[col] = mls1_X[col].fillna(mls1_X[col].mode()[0] if not mls1_X[col].mode().empty else 'Unknown')
                        
                        mls1_X_processed = pd.get_dummies(mls1_X, drop_first=True, dummy_na=False)

                        if mls1_X_processed.empty or mls1_y.empty:
                            st.error("Not enough data after preprocessing for model training.")
                        else:
                            # Ensure stratify is only used if there are at least 2 classes.
                            # train_test_split itself will raise ValueError if a class has < 2 samples for stratification.
                            stratify_param = mls1_y if mls1_y.nunique() >= 2 else None
                            X_train, X_test, y_train, y_test = train_test_split(mls1_X_processed, mls1_y, test_size=0.3, random_state=42, stratify=stratify_param)
                            
                            model_rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
                            model_rf.fit(X_train, y_train)
                            y_pred = model_rf.predict(X_test)

                            st.markdown("##### Model Performance (Random Forest Classifier)")
                            st.text("Classification Report:")
                            st.text(classification_report(y_test, y_pred, zero_division=0))

                            st.text("Confusion Matrix:")
                            cm_fig, ax_cm = plt.subplots()
                            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                            ax_cm.set_xlabel('Predicted')
                            ax_cm.set_ylabel('Actual')
                            st.pyplot(cm_fig)

                            st.markdown("###### Feature Importances")
                            importances = pd.Series(model_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                            st.bar_chart(importances.head(15)) # Show top 15 features
                            st.dataframe(importances.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).head(20))

                    except Exception as e:
                        st.error(f"An error occurred during model training: {e}")
                        st.error("Ensure your target column is binary (e.g., True/False, 0/1, or two distinct categories like 'Yes'/'No').")
        with st.expander("🔬 Principal Component Analysis (PCA)", expanded=False): # This is still part of tab3
            st.info("Reduce dimensionality of numeric features and visualize data in lower dimensions. Select multiple numeric columns for PCA.")
            numeric_cols_pca = get_numeric_columns(df)

            if len(numeric_cols_pca) < 2:
                st.warning("PCA requires at least two numeric columns.")
            else:
                selected_features_pca = st.multiselect(
                    "Select numeric features for PCA:",
                    numeric_cols_pca,
                    default=[col for col in ['Amount', 'Qty', 'shipping fee', 'Courier Charges'] if col in numeric_cols_pca][:min(len(numeric_cols_pca), 4)], # Sensible defaults
                    key="pca_features"
                )
                
                # Determine n_components_pca and conditionally show slider
                n_components_pca = 2  # Default value

                if selected_features_pca and len(selected_features_pca) >= 2:
                    min_slider_val = 2
                    # Max components is limited by number of features or an arbitrary cap like 10
                    max_slider_val = min(len(selected_features_pca), 10)

                    if max_slider_val < min_slider_val:
                        # This case implies len(selected_features_pca) < 2, which should be caught by button logic.
                        # n_components_pca remains 2.
                        st.warning(f"Not enough features selected ({len(selected_features_pca)}) for PCA component selection. Minimum is {min_slider_val}.")
                    elif max_slider_val == min_slider_val:
                        n_components_pca = min_slider_val
                        st.info(f"Number of principal components is set to {n_components_pca} (minimum required and maximum possible with {len(selected_features_pca)} selected features).")
                    else: # max_slider_val > min_slider_val, so slider is valid
                        n_components_pca = st.slider(
                            "Number of Principal Components to compute:",
                            min_value=min_slider_val,
                            max_value=max_slider_val,
                            value=min_slider_val,  # Default to the minimum possible (e.g., 2)
                            key="pca_n_components",
                            help="Must be less than or equal to the number of selected features."
                        )
                # If selected_features_pca is empty or has < 2 features, n_components_pca remains 2.
                # The button click logic will handle warnings for insufficient features.

                if st.button("✨ Run PCA", key="pca_run"):
                    if not selected_features_pca or len(selected_features_pca) < 2:
                        st.warning("Please select at least two numeric features for PCA.")
                    elif n_components_pca > len(selected_features_pca):
                        st.warning("Number of components cannot exceed the number of selected features.")
                    else:
                        try:
                            pca_df_features = df[selected_features_pca].copy().dropna()
                            if pca_df_features.empty or len(pca_df_features) < n_components_pca:
                                st.warning("Not enough data after dropping NaNs for selected features, or fewer data points than components.")
                            else:
                                scaler_pca = StandardScaler()
                                scaled_features_pca = scaler_pca.fit_transform(pca_df_features)

                                pca = PCA(n_components=n_components_pca, random_state=42)
                                principal_components = pca.fit_transform(scaled_features_pca)
                                
                                pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components_pca)])

                                st.markdown("##### PCA Results")
                                st.write(f"Explained Variance Ratio by each component: {pca.explained_variance_ratio_}")
                                st.write(f"Total Explained Variance by {n_components_pca} components: {pca.explained_variance_ratio_.sum():.4f}")

                                st.markdown("###### Scree Plot (Explained Variance)")
                                fig_scree, ax_scree = plt.subplots()
                                ax_scree.bar(range(1, n_components_pca + 1), pca.explained_variance_ratio_, alpha=0.7, align='center', label='Individual explained variance')
                                ax_scree.step(range(1, n_components_pca + 1), pca.explained_variance_ratio_.cumsum(), where='mid', label='Cumulative explained variance')
                                ax_scree.set_ylabel('Explained variance ratio')
                                ax_scree.set_xlabel('Principal component index')
                                ax_scree.set_xticks(range(1, n_components_pca + 1))
                                ax_scree.legend(loc='best')
                                plt.tight_layout()
                                st.pyplot(fig_scree)

                                if n_components_pca >= 2:
                                    st.markdown("###### Scatter Plot of First Two Principal Components")
                                    fig_pc_scatter, ax_pc_scatter = plt.subplots()
                                    sns.scatterplot(x=pc_df['PC1'], y=pc_df['PC2'], ax=ax_pc_scatter, alpha=0.5)
                                    ax_pc_scatter.set_xlabel('Principal Component 1')
                                    ax_pc_scatter.set_ylabel('Principal Component 2')
                                    ax_pc_scatter.set_title('First Two Principal Components')
                                    st.pyplot(fig_pc_scatter)
                                
                                st.write("Principal Components Data (first 100 rows):")
                                st.dataframe(pc_df.head(100))

                        except Exception as e:
                            st.error(f"Error during PCA: {e}")
        with st.expander("📈 Predictive Sales Regression", expanded=False): # This is still part of tab3
            st.info("Train a regression model (e.g., RandomForestRegressor) to predict a continuous sales metric like 'Amount' or 'Qty'.")
            
            all_cols_psr = df.columns.tolist()
            numeric_cols_psr = get_numeric_columns(df)
            categorical_features_psr = get_categorical_columns(df, nunique_threshold=50)

            target_col_psr = st.selectbox(
                "Select Target Column (Numeric, e.g., 'Amount', 'Qty'):",
                numeric_cols_psr,
                index=numeric_cols_psr.index('Amount') if 'Amount' in numeric_cols_psr else 0,
                key="psr_target"
            )
            available_features_psr = [col for col in numeric_cols_psr + categorical_features_psr if col != target_col_psr]
            default_features_psr = [f for f in ['Qty', 'Category', 'Sales Channel', 'Fulfilment', 'ship-state', 'B2B'] if f in available_features_psr and f != target_col_psr]
            
            selected_features_psr = st.multiselect(
                "Select Feature Columns:",
                available_features_psr,
                default=default_features_psr,
                key="psr_features"
            )

            if st.button("🎯 Train Sales Regression Model", key="psr_run"):
                if not target_col_psr or not selected_features_psr:
                    st.warning("Please select a target column and at least one feature column.")
                else:
                    try:
                        psr_X = df[selected_features_psr].copy()
                        psr_y = df[target_col_psr].copy()

                        # Align X and y by dropping NaNs from target first, then from features
                        psr_y = psr_y.dropna()
                        psr_X = psr_X.loc[psr_y.index]

                        # Preprocessing: Impute NaNs and One-Hot Encode features
                        for col in psr_X.select_dtypes(include=np.number).columns:
                            psr_X[col] = psr_X[col].fillna(psr_X[col].median())
                        for col in psr_X.select_dtypes(include='object').columns:
                            psr_X[col] = psr_X[col].fillna(psr_X[col].mode()[0] if not psr_X[col].mode().empty else 'Unknown')
                        
                        psr_X_processed = pd.get_dummies(psr_X, drop_first=True, dummy_na=False)
                        
                        # Final alignment after get_dummies might change index if dummy_na=True was used and created NaNs
                        # However, with dummy_na=False, index should be preserved.
                        # Still, good practice to ensure alignment before split.
                        common_index = psr_X_processed.index.intersection(psr_y.index)
                        psr_X_processed = psr_X_processed.loc[common_index]
                        psr_y = psr_y.loc[common_index]

                        if psr_X_processed.empty or psr_y.empty:
                            st.error("Not enough data after preprocessing for model training.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(psr_X_processed, psr_y, test_size=0.3, random_state=42)
                            
                            model_rfr = RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
                            model_rfr.fit(X_train, y_train)
                            y_pred = model_rfr.predict(X_test)

                            st.markdown("##### Model Performance (Random Forest Regressor)")
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                            st.metric("R-squared (R²)", f"{r2:.2f}")

                            st.markdown("###### Feature Importances")
                            importances_psr = pd.Series(model_rfr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                            st.bar_chart(importances_psr.head(15))
                            st.dataframe(importances_psr.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).head(20))

                            st.markdown("###### Actual vs. Predicted Plot (Sample)")
                            fig_pred, ax_pred = plt.subplots()
                            sample_size_plot = min(len(y_test), 200) # Plot a sample for performance
                            sns.scatterplot(x=y_test[:sample_size_plot], y=y_pred[:sample_size_plot], ax=ax_pred, alpha=0.6)
                            ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
                            ax_pred.set_xlabel("Actual Values")
                            ax_pred.set_ylabel("Predicted Values")
                            ax_pred.set_title(f"Actual vs. Predicted {target_col_psr} (Sample)")
                            st.pyplot(fig_pred)

                    except Exception as e:
                        st.error(f"An error occurred during regression model training: {e}")
        with st.expander("🔄 Inventory Turnover/Velocity Analysis", expanded=False): # This is still part of tab3
            st.info("Analyze sales velocity (units sold per period). True inventory turnover requires average inventory data, which is not in this dataset. This tool focuses on sales velocity.")
            
            all_cols_itv = df.columns.tolist()
            numeric_cols_itv = get_numeric_columns(df)
            date_cols_itv = date_cols

            sku_col_itv = st.selectbox("Select Product ID/SKU column:", all_cols_itv, index=all_cols_itv.index('SKU') if 'SKU' in all_cols_itv else 0, key="itv_sku")
            qty_col_itv = st.selectbox("Select Quantity Sold column:", numeric_cols_itv, index=numeric_cols_itv.index('Qty') if 'Qty' in numeric_cols_itv else 0, key="itv_qty")
            date_col_itv = st.selectbox("Select Date column:", date_cols_itv, index=date_cols_itv.index('Date') if 'Date' in date_cols_itv else 0, key="itv_date")
            time_period_itv = st.selectbox("Aggregation period for velocity:", ["D", "W", "M"], index=1, format_func=lambda x: {"D":"Daily", "W":"Weekly", "M":"Monthly"}[x], key="itv_period")
            top_n_itv = st.slider("Number of top/bottom products by velocity to show:", 5, 20, 10, key="itv_top_n")

            if st.button("💨 Analyze Sales Velocity", key="itv_run"):
                if not all([sku_col_itv, qty_col_itv, date_col_itv]):
                    st.warning("Please select SKU, Quantity, and Date columns.")
                else:
                    try:
                        itv_df = df[[sku_col_itv, qty_col_itv, date_col_itv]].copy()
                        itv_df[date_col_itv] = pd.to_datetime(itv_df[date_col_itv], errors='coerce')
                        itv_df = itv_df.dropna()

                        if itv_df.empty:
                            st.warning("No data available for sales velocity analysis.")
                        else:
                            product_days = itv_df.groupby(sku_col_itv)[date_col_itv].agg(['min', 'max'])
                            product_days['duration_days'] = (product_days['max'] - product_days['min']).dt.days + 1
                            product_total_qty = itv_df.groupby(sku_col_itv)[qty_col_itv].sum()
                            
                            velocity_summary_df = pd.concat([product_total_qty, product_days], axis=1).rename(columns={qty_col_itv: 'TotalQtySold'})
                            
                            if time_period_itv == "D":
                                velocity_summary_df['SalesVelocity'] = velocity_summary_df['TotalQtySold'] / velocity_summary_df['duration_days']
                                velocity_unit_itv = "per day"
                            elif time_period_itv == "W":
                                velocity_summary_df['SalesVelocity'] = velocity_summary_df['TotalQtySold'] / (velocity_summary_df['duration_days'] / 7)
                                velocity_unit_itv = "per week"
                            else: # Monthly
                                velocity_summary_df['SalesVelocity'] = velocity_summary_df['TotalQtySold'] / (velocity_summary_df['duration_days'] / 30.44)
                                velocity_unit_itv = "per month"
                            
                            velocity_summary_df = velocity_summary_df.replace([np.inf, -np.inf], 0).fillna(0)

                            st.markdown(f"##### Product Sales Velocity ({velocity_unit_itv})")
                            st.markdown(f"###### Top {top_n_itv} Fast-Moving Products")
                            st.dataframe(velocity_summary_df.sort_values(by='SalesVelocity', ascending=False).head(top_n_itv))
                            st.markdown(f"###### Top {top_n_itv} Slow-Moving Products (Velocity > 0)")
                            st.dataframe(velocity_summary_df[velocity_summary_df['SalesVelocity'] > 0].sort_values(by='SalesVelocity', ascending=True).head(top_n_itv))
                    except Exception as e:
                        st.error(f"Error during Sales Velocity analysis: {e}")
        with st.expander("👥 Customer F-M Segmentation (Frequency-Monetary)", expanded=False): # This is the last expander in tab3
            st.info("Segment customers/entities based on purchase Frequency (number of orders) and Monetary value (total spend). Uses 'Order ID' as a proxy for customer if no dedicated Customer ID is available.")
            
            all_cols_fm = df.columns.tolist()
            numeric_cols_fm = get_numeric_columns(df)
            # 'Order ID' is used as the entity identifier here.
            
            entity_id_col_fm = st.selectbox("Select Customer/Entity ID column (e.g., 'Order ID'):", all_cols_fm, index=all_cols_fm.index('Order ID') if 'Order ID' in all_cols_fm else 0, key="fm_entity_id")
            amount_col_fm = st.selectbox("Select Total Amount column:", numeric_cols_fm, index=numeric_cols_fm.index('Amount') if 'Amount' in numeric_cols_fm else 0, key="fm_amount")
            # For frequency, we count distinct orders. If Order ID is unique per item, we count occurrences of entity_id_col_fm.
            # If Order ID represents a transaction, we count unique Order IDs per entity_id_col_fm.
            # Given the dataset, Order ID is unique per row. So, if entity_id_col_fm is also Order ID, frequency will be 1 for all.
            # This tool is more meaningful if entity_id_col_fm is a true Customer ID that can have multiple Order IDs.
            # For demonstration, we'll proceed assuming entity_id_col_fm can group transactions.

            n_clusters_fm = st.slider("Number of F-M segments:", 2, 8, 3, key="fm_n_clusters")

            if st.button("🫂 Run F-M Segmentation", key="fm_run"):
                if not entity_id_col_fm or not amount_col_fm:
                    st.warning("Please select Entity ID and Amount columns.")
                else:
                    try:
                        # Aggregate by the chosen entity ID
                        # Frequency: count of unique orders (if Order ID is distinct per order) or count of transactions
                        # Monetary: sum of amount
                        # For this dataset, Order ID is unique per row. If entity_id_col_fm is 'Order ID', frequency will be 1.
                        # If a different column is chosen for entity_id_col_fm, then we count Order IDs under it.
                        
                        fm_agg_df = df.groupby(entity_id_col_fm).agg(
                            Frequency=(entity_id_col_fm, 'count'), # Number of transactions/items for this entity
                            Monetary=(amount_col_fm, 'sum')
                        ).reset_index()

                        if fm_agg_df.empty or len(fm_agg_df) < n_clusters_fm:
                            st.warning(f"Not enough unique entities or data points ({len(fm_agg_df)}) for {n_clusters_fm} F-M segments.")
                        else:
                            scaler_fm = StandardScaler()
                            fm_scaled = scaler_fm.fit_transform(fm_agg_df[['Frequency', 'Monetary']])
                            
                            kmeans_fm = KMeans(n_clusters=n_clusters_fm, random_state=42, n_init='auto')
                            fm_agg_df['Segment'] = kmeans_fm.fit_predict(fm_scaled)

                            st.markdown("##### F-M Segmentation Results")
                            st.write("Segment Profiles (Mean Frequency & Monetary):")
                            st.dataframe(fm_agg_df.groupby('Segment')[['Frequency', 'Monetary']].mean())
                            st.write("Segment Sizes:")
                            st.dataframe(fm_agg_df['Segment'].value_counts().sort_index())

                            fig_fm_scatter, ax_fm_scatter = plt.subplots()
                            sns.scatterplot(data=fm_agg_df, x='Frequency', y='Monetary', hue='Segment', palette='viridis', ax=ax_fm_scatter, s=30, alpha=0.7)
                            ax_fm_scatter.set_title('Customer Segments by Frequency and Monetary Value')
                            ax_fm_scatter.set_xlabel('Frequency (Number of Transactions/Items)')
                            ax_fm_scatter.set_ylabel('Monetary Value (Total Spend)')
                            st.pyplot(fig_fm_scatter)
                            st.caption(f"Note: If '{entity_id_col_fm}' is 'Order ID' (unique per row), 'Frequency' will be 1 for all, and segmentation might not be meaningful.")
                    except Exception as e:
                        st.error(f"Error during F-M Segmentation: {e}")
    # End of with tab3. New tabs start here at the same indentation level.

    with tab4:
        st.header("🔍 Data Deep Dive & Quality Assurance")
        st.write(f"Perform in-depth exploratory data analysis (EDA), assess data quality, and profile your '{DATASET_FILENAME}' dataset.")
        st.info("This section is under development. Future tools could include: \n"
                "- Detailed column statistics (missing values, unique counts, distributions)\n"
                "- Data type validation and correction suggestions\n"
                "- Outlier detection summaries beyond Isolation Forest\n"
                "- Correlation matrix heatmaps with advanced options\n"
                "- Data profiling reports (e.g., using pandas-profiling if integrated)")

    with tab5:
        st.header("🔧 Feature Engineering & Preprocessing")
        st.write("Create new features, transform existing ones, and prepare your data for machine learning models.")
        st.info("This section is under development. Future tools could include: \n"
                "- Date feature extraction (e.g., day of week, month, quarter, cyclical features)\n"
                "- Text feature extraction (e.g., TF-IDF, word counts from product descriptions if available)\n"
                "- Binning/Discretization of numerical features\n"
                "- Interaction feature creation (e.g., price * quantity)\n"
                "- Advanced scaling and normalization techniques\n"
                "- Feature selection methods (e.g., based on variance, correlation, model-based selection)")

    with tab6:
        st.header("🧠 Advanced Modeling Lab")
        st.write("Experiment with more sophisticated machine learning models and techniques.")
        st.info("This section is under development. Future tools could include: \n"
                "- Advanced Regression Models (e.g., XGBoost, LightGBM, Neural Networks for regression)\n"
                "- Advanced Classification Models (e.g., SVM, Gradient Boosting, Naive Bayes)\n"
                "- Clustering Algorithms beyond K-Means (e.g., DBSCAN, Hierarchical Clustering)\n"
                "- Time Series Forecasting Models (e.g., Prophet, SARIMA with auto-tuning)\n"
                "- Model comparison and hyperparameter tuning frameworks")

    with tab7:
        st.header("📈 Interactive Dashboards & Export")
        st.write("Create dynamic visualizations and export data, results, or reports.")
        st.info("This section is under development. Future tools could include: \n"
                "- Customizable Plotly Express charts for deeper visual exploration\n"
                "- Interactive pivot tables and cross-tabulations\n"
                "- Options to export filtered/processed data to CSV/Excel\n"
                "- Functionality to save generated charts or tables as images/PDFs\n"
                "- Basic report generation based on selected analyses")
except FileNotFoundError:
    st.error(f"🚨 Error: `{DATASET_FILENAME}` not found. Please make sure the file is in the same directory as `app.py`.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error(f"🚨 Error: `{DATASET_FILENAME}` is empty. Please provide a valid CSV file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading or initial setup: {e}")
    st.stop()
