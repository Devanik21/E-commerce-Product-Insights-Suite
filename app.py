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
            df[date_column_to_format] = pd.to_datetime(df[date_column_to_format], errors='coerce', errors='ignore')

    other_date_cols = [col for col in df.columns if 'date' in col.lower() and col != date_column_to_format]
    for col in other_date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', errors='ignore')

    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist() # Update global list

    # --- Sidebar for API Key and AI Model Info ---
    st.sidebar.subheader("✨ AI Configuration")
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", help="Get your API key from Google AI Studio.")
    st.sidebar.caption("Using AI Model: Gemini 1.5 Flash (via `gemini-1.5-flash-latest`)")
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Analysis Modules")

    # --- Main content with Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Traditional Analysis",
        "🤖 AI Powered Insights",
        "🔬 Advanced Analytics Toolkit"
    ])
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

    with tab3:
        st.header("🔬 Advanced Analytics Toolkit")
        st.write("Explore a range of advanced analytical techniques. Select a category to see available tools.")
        st.caption("Note: Most tools listed here are conceptual placeholders for advanced analyses. Implementation would require specific data, model development, and potentially different datasets than the preloaded one.")

        # Category 1: Advanced Statistical Modeling (ASM)
        with st.expander("📈 Advanced Statistical Modeling (ASM)"):
            st.write("Implementations of advanced statistical models. Ensure your dataset has appropriate columns for each analysis.")

            # --- Helper functions for column selection ---
            def get_numeric_columns(data_frame):
                return data_frame.select_dtypes(include=np.number).columns.tolist()

            def get_categorical_columns(data_frame, nunique_threshold=30):
                return [col for col in data_frame.columns if data_frame[col].nunique() < nunique_threshold and (data_frame[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data_frame[col]))]

            numeric_cols = get_numeric_columns(df)
            categorical_cols = get_categorical_columns(df)
            # date_cols is already defined globally and populated

            # --- ASM 1: Advanced Hypothesis Testing ---
            st.subheader("ASM 1: Advanced Hypothesis Testing")
            test_type = st.selectbox("Select Hypothesis Test", ["Chi-Squared Test", "ANOVA", "Kruskal-Wallis Test"], key="asm_ht_type")

            if test_type == "Chi-Squared Test":
                st.markdown("Tests for independence between two categorical variables.")
                if len(categorical_cols) >= 2:
                    cat_col1 = st.selectbox("Select first categorical variable:", categorical_cols, key="asm_chi_cat1")
                    cat_col2 = st.selectbox("Select second categorical variable:", [c for c in categorical_cols if c != cat_col1], key="asm_chi_cat2")
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
                if categorical_cols and numeric_cols:
                    cat_col_anova = st.selectbox("Select categorical grouping variable:", categorical_cols, key="asm_anova_cat")
                    num_col_anova = st.selectbox("Select numerical variable:", numeric_cols, key="asm_anova_num")
                    if cat_col_anova and num_col_anova and st.button("Run ANOVA", key="asm_anova_run"):
                        try:
                            groups = [df[num_col_anova][df[cat_col_anova] == group].dropna() for group in df[cat_col_anova].unique()]
                            groups = [g for g in groups if len(g) > 1] # Ensure groups have enough data
                            if len(groups) < 2:
                                st.warning("Need at least two groups with sufficient data for ANOVA.")
                            else:
                                f_stat, p_val = f_oneway(*groups)
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
                if categorical_cols and numeric_cols:
                    cat_col_kw = st.selectbox("Select categorical grouping variable:", categorical_cols, key="asm_kw_cat")
                    num_col_kw = st.selectbox("Select numerical variable:", numeric_cols, key="asm_kw_num")
                    if cat_col_kw and num_col_kw and st.button("Run Kruskal-Wallis Test", key="asm_kw_run"):
                        try:
                            groups = [df[num_col_kw][df[cat_col_kw] == group].dropna() for group in df[cat_col_kw].unique()]
                            groups = [g for g in groups if len(g) > 0]
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
            st.markdown("Model binary outcomes (e.g., predict high/low discount) based on product features.")
            if numeric_cols or categorical_cols:
                # Target variable creation
                st.write("Define Target Variable (Binary):")
                target_source_col = st.selectbox("Select column to create binary target from:", numeric_cols + categorical_cols, key="asm_logreg_target_src")
                
                target_col_name = "logistic_target"
                df_logreg = df.copy()

                if target_source_col:
                    if target_source_col in numeric_cols:
                        threshold = st.number_input(f"Enter threshold for '{target_source_col}' to define 1 (e.g., > threshold)", value=df_logreg[target_source_col].median() if not df_logreg[target_source_col].empty else 0, key="asm_logreg_thresh")
                        df_logreg[target_col_name] = (df_logreg[target_source_col] > threshold).astype(int)
                    else: # Categorical source
                        positive_class = st.selectbox(f"Select the 'positive' class (1) for '{target_source_col}':", df_logreg[target_source_col].unique(), key="asm_logreg_pos_class")
                        df_logreg[target_col_name] = (df_logreg[target_source_col] == positive_class).astype(int)
                    
                    st.write(f"Target variable '{target_col_name}' created. Value counts:")
                    st.write(df_logreg[target_col_name].value_counts())

                    feature_cols_options = [col for col in numeric_cols + categorical_cols if col != target_source_col]
                    selected_features = st.multiselect("Select feature variables:", feature_cols_options, key="asm_logreg_features")

                    if selected_features and st.button("Run Logistic Regression", key="asm_logreg_run"):
                        try:
                            X = df_logreg[selected_features]
                            y = df_logreg[target_col_name]

                            # Handle categorical features with one-hot encoding
                            X = pd.get_dummies(X, drop_first=True)
                            X = sm.add_constant(X) # Add intercept

                            if X.empty or y.empty or len(X) != len(y):
                                st.error("Feature set or target variable is empty or mismatched after preprocessing.")
                            elif y.nunique() < 2:
                                st.error("Target variable must have at least two unique classes for logistic regression.")
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None)
                                
                                log_reg_model = sm.Logit(y_train, X_train).fit(disp=0) # disp=0 suppresses convergence messages
                                st.subheader("Logistic Regression Results")
                                st.text(log_reg_model.summary())

                                y_pred_proba = log_reg_model.predict(X_test)
                                y_pred = (y_pred_proba > 0.5).astype(int)

                                st.subheader("Model Evaluation (Test Set)")
                                st.text(classification_report(y_test, y_pred))
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
            if not date_cols:
                st.warning("No date columns found in the dataset. Time series analysis requires a date column.")
            elif not numeric_cols:
                st.warning("No numeric columns found for time series values.")
            else:
                time_col = st.selectbox("Select Date/Time column:", date_cols, key="asm_ts_datecol")
                value_col = st.selectbox("Select Value column for time series:", numeric_cols, key="asm_ts_valcol")
                aggregation_freq = st.selectbox("Aggregate data by:", ["D", "W", "M", "Q", "Y"], index=2, key="asm_ts_freq", help="D: Day, W: Week, M: Month, Q: Quarter, Y: Year")
                
                if time_col and value_col and st.button("Analyze Time Series", key="asm_ts_run"):
                    try:
                        ts_df = df[[time_col, value_col]].copy()
                        ts_df[time_col] = pd.to_datetime(ts_df[time_col], errors='coerce')
                        ts_df = ts_df.dropna(subset=[time_col, value_col])
                        ts_df = ts_df.set_index(time_col)
                        
                        # Aggregate data
                        ts_aggregated = ts_df[value_col].resample(aggregation_freq).mean() # Can change to .sum() or other agg
                        ts_aggregated = ts_aggregated.dropna()

                        if len(ts_aggregated) < 12 : # Minimum for seasonal decomposition with yearly seasonality
                             st.warning(f"Not enough data points ({len(ts_aggregated)}) after aggregation for meaningful decomposition or ARIMA. Need at least ~12 for monthly data over a year.")
                        else:
                            st.write(f"Aggregated Time Series (first 5 rows, {aggregation_freq} frequency):")
                            st.write(ts_aggregated.head())

                            # Decomposition
                            st.subheader("Time Series Decomposition")
                            decomposition = seasonal_decompose(ts_aggregated, model='additive', period=max(1, min(12, len(ts_aggregated)//2))) # Heuristic for period
                            fig_decompose = decomposition.plot()
                            fig_decompose.set_size_inches(10, 8)
                            st.pyplot(fig_decompose)

                            # ARIMA Model (simple example)
                            st.subheader("ARIMA Model (Example)")
                            if len(ts_aggregated) >= 20: # Basic check for ARIMA
                                try:
                                    # A simple ARIMA(p,d,q) order, can be optimized with ACF/PACF plots or auto_arima
                                    model_arima = ARIMA(ts_aggregated, order=(5,1,0)).fit()
                                    st.text(model_arima.summary())
                                    
                                    # Forecast
                                    forecast_steps = min(12, len(ts_aggregated)//4)
                                    forecast = model_arima.get_forecast(steps=forecast_steps)
                                    forecast_df = forecast.summary_frame()

                                    fig_arima, ax = plt.subplots(figsize=(10, 6))
                                    ts_aggregated.plot(ax=ax, label='Observed')
                                    forecast_df['mean'].plot(ax=ax, label='Forecast')
                                    ax.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='k', alpha=.15)
                                    ax.set_title(f'ARIMA Forecast for {value_col}')
                                    ax.legend()
                                    st.pyplot(fig_arima)
                                except Exception as e_arima:
                                    st.error(f"Error fitting/forecasting with ARIMA: {e_arima}. Try different ARIMA orders or ensure stationary data.")
                            else:
                                st.warning("Not enough data points for a reliable ARIMA model example after aggregation.")
                    except Exception as e:
                        st.error(f"Error in Time Series Analysis: {e}")

            st.markdown("---")
            # --- ASM 4: Survival Analysis (Conceptual Example) ---
            st.subheader("ASM 4: Survival Analysis (Conceptual Example)")
            st.markdown("""
            Survival analysis studies the time until an event occurs. 
            Since the current dataset might not have direct duration/event columns (e.g., product lifecycle, time-to-churn),
            this is a conceptual demonstration using synthetically generated data.
            """)
            if st.button("Show Survival Analysis Example", key="asm_sa_run"):
                try:
                    # Generate synthetic data: T = time to event, E = event occurred (1) or censored (0)
                    np.random.seed(42)
                    N = 200
                    T = np.random.exponential(scale=10, size=N) + np.random.normal(loc=5, scale=2, size=N)
                    T = np.clip(T, 1, 50) # Ensure positive durations
                    E = np.random.binomial(1, 0.7, size=N) # 70% experience the event
                    
                    # Censor some data for those who didn't experience the event
                    censoring_time = np.random.uniform(5, 40, size=N)
                    T[E==0] = np.minimum(T[E==0], censoring_time[E==0])

                    st.write("Synthetic Data Sample (First 10 entries):")
                    st.write(pd.DataFrame({'Duration (T)': T, 'EventOccurred (E)': E}).head(10))

                    kmf = KaplanMeierFitter()
                    kmf.fit(T, event_observed=E)

                    fig_km, ax = plt.subplots(figsize=(8,6))
                    kmf.plot_survival_function(ax=ax)
                    ax.set_title('Kaplan-Meier Survival Curve (Synthetic Data)')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Survival Probability')
                    st.pyplot(fig_km)

                    st.write("Median Survival Time (where survival probability is 0.5):")
                    st.write(f"{kmf.median_survival_time_:.2f} time units")
                except Exception as e:
                    st.error(f"Error in Survival Analysis example: {e}")

            st.markdown("---")
            # --- ASM 5: Bayesian A/B Testing ---
            st.subheader("ASM 5: Bayesian A/B Testing")
            st.markdown("Compare two groups (A/B) using Bayesian methods to estimate the probability of one being better.")
            if 'Group' in df.columns and 'Conversion' in df.columns:
                group_col_ab = 'Group'
                conversion_col_ab = 'Conversion' # Assumed to be 0 or 1

                if df[conversion_col_ab].nunique() == 2 and set(df[conversion_col_ab].unique()) <= {0, 1}:
                    if st.button("Run Bayesian A/B Test", key="asm_bab_run"):
                        try:
                            summary = df.groupby(group_col_ab)[conversion_col_ab].agg(['sum', 'count'])
                            summary.columns = ['conversions', 'total_users']
                            st.write("A/B Test Data Summary:")
                            st.write(summary)

                            if len(summary) == 2: # Assuming two groups
                                conversions_a = summary['conversions'].iloc[0]
                                total_a = summary['total_users'].iloc[0]
                                conversions_b = summary['conversions'].iloc[1]
                                total_b = summary['total_users'].iloc[1]

                                with pm.Model() as bayesian_ab_model:
                                    # Priors for conversion rates (Beta distribution is common for probabilities)
                                    p_A = pm.Beta('p_A', alpha=1, beta=1) # Uniform prior
                                    p_B = pm.Beta('p_B', alpha=1, beta=1)

                                    # Likelihoods (Binomial distribution for number of conversions)
                                    obs_A = pm.Binomial('obs_A', n=total_a, p=p_A, observed=conversions_a)
                                    obs_B = pm.Binomial('obs_B', n=total_b, p=p_B, observed=conversions_b)

                                    # Difference between conversion rates
                                    delta = pm.Deterministic('delta', p_B - p_A)
                                    
                                    # Relative uplift
                                    uplift = pm.Deterministic('uplift', (p_B - p_A) / p_A)

                                    trace = pm.sample(2000, tune=1000, cores=1, progressbar=True) # cores=1 for streamlit compatibility

                                st.subheader("Bayesian A/B Test Results")
                                fig_trace, ax_trace = plt.subplots(figsize=(10,6))
                                az.plot_posterior(trace, var_names=['p_A', 'p_B', 'delta', 'uplift'], ax=ax_trace)
                                plt.tight_layout()
                                st.pyplot(fig_trace)

                                prob_b_better_a = (trace.posterior['delta'].values > 0).mean()
                                st.write(f"Probability that Group B's conversion rate is greater than Group A's: {prob_b_better_a:.2%}")

                                if prob_b_better_a > 0.95:
                                    st.success("There is strong evidence that Group B performs better than Group A.")
                                elif prob_b_better_a < 0.05:
                                    st.success("There is strong evidence that Group A performs better than Group B.")
                                else:
                                    st.info("The evidence is not strong enough to definitively conclude one group is better than the other.")
                            else:
                                st.warning("Bayesian A/B testing requires exactly two groups in the 'Group' column.")
                        except Exception as e:
                            st.error(f"Error running Bayesian A/B Test: {e}")
                else:
                    st.warning(f"The 'Conversion' column ('{conversion_col_ab}') must be binary (0 or 1) for Bayesian A/B testing.")
            else:
                st.warning("Bayesian A/B testing requires 'Group' and 'Conversion' columns in the dataset.")
        
        # Category 2: Machine Learning - Supervised (MLS)
        with st.expander("🤖 Machine Learning - Supervised (MLS)"):
            st.subheader("MLS 1: Support Vector Machines (SVM)")
            st.markdown("Powerful classification and regression technique, effective in high-dimensional spaces and for non-linear relationships.")
            st.subheader("MLS 2: Gradient Boosting Machines (XGBoost, LightGBM, CatBoost)")
            st.markdown("Advanced ensemble methods for high-performance classification and regression tasks, handling complex data patterns.")
            st.subheader("MLS 3: Ensemble Learning - Stacking & Voting")
            st.markdown("Combine predictions from multiple diverse models to improve overall predictive accuracy and robustness.")
            st.subheader("MLS 4: Model Interpretability (SHAP, LIME)")
            st.markdown("Understand and explain the predictions of complex machine learning models, identifying key feature contributions.")
            st.subheader("MLS 5: Automated Machine Learning (AutoML)")
            st.markdown("Automate the end-to-end process of applying machine learning, from data preprocessing to model selection and hyperparameter tuning.")

        # Category 3: Machine Learning - Unsupervised (MLU)
        with st.expander("🔍 Machine Learning - Unsupervised (MLU)"):
            st.subheader("MLU 1: DBSCAN & OPTICS Clustering")
            st.markdown("Density-based clustering algorithms capable of finding arbitrarily shaped clusters and identifying noise points.")
            st.subheader("MLU 2: Hierarchical Clustering")
            st.markdown("Build a hierarchy of clusters, allowing for exploration of data structure at different levels of granularity.")
            st.subheader("MLU 3: Advanced Dimensionality Reduction (PCA, t-SNE, UMAP)")
            st.markdown("Reduce the number of features while preserving important information, for visualization or as a preprocessing step.")
            st.subheader("MLU 4: Advanced Anomaly Detection")
            st.markdown("Identify unusual data points or patterns using techniques like Isolation Forest, One-Class SVM, or Autoencoders.")
            st.subheader("MLU 5: Association Rule Mining (Apriori, FP-Growth)")
            st.markdown("Discover interesting relationships and patterns in large datasets, such as 'frequently bought together' items.")

        # Category 4: Natural Language Processing & Text Analytics (NLP)
        with st.expander("📜 Natural Language Processing & Text Analytics (NLP)"):
            st.subheader("NLP 1: Advanced Sentiment Analysis")
            st.markdown("Perform aspect-based sentiment analysis or use transformer models for nuanced understanding of text sentiment in product details or reviews.")
            st.subheader("NLP 2: Topic Modeling (LDA, NMF, BERTopic)")
            st.markdown("Discover latent topics within large collections of text documents, such as product descriptions or customer feedback.")
            st.subheader("NLP 3: Named Entity Recognition (NER)")
            st.markdown("Identify and categorize key entities (like brands, materials, features) within product details or other textual data.")
            st.subheader("NLP 4: Text Summarization")
            st.markdown("Automatically generate concise summaries of long product descriptions, reviews, or articles using extractive or abstractive methods.")
            st.subheader("NLP 5: Semantic Search & Question Answering")
            st.markdown("Implement search systems that understand the meaning behind queries, or build systems to answer questions based on product documentation.")

        # Category 5: Deep Learning Applications (DLA)
        with st.expander("🧠 Deep Learning Applications (DLA)"):
            st.subheader("DLA 1: Image-Based Product Analysis")
            st.markdown("Use Convolutional Neural Networks (CNNs) for product categorization, feature extraction, or visual search from product images.")
            st.subheader("DLA 2: Sequential Data Modeling (RNNs, LSTMs, Transformers)")
            st.markdown("Analyze sequences like customer journey paths, clickstreams, or time-series sales data for prediction or pattern recognition.")
            st.subheader("DLA 3: Advanced Recommendation Systems")
            st.markdown("Build sophisticated product recommenders using deep learning techniques like collaborative filtering with neural networks or content-based filtering with embeddings.")
            st.subheader("DLA 4: Generative Models for Content Creation")
            st.markdown("Explore using GANs or other generative models for creating synthetic product images, designs, or marketing copy (requires specialized data and setup).")
            st.subheader("DLA 5: Fraud Detection with Deep Learning")
            st.markdown("Apply deep learning models to detect complex fraudulent patterns in transaction data or user behavior.")

        # Category 6: Network & Graph Analysis (NGA)
        with st.expander("🕸️ Network & Graph Analysis (NGA)"):
            st.subheader("NGA 1: Product Co-occurrence & Association Networks")
            st.markdown("Visualize and analyze relationships between products (e.g., frequently bought together) or product features.")
            st.subheader("NGA 2: Customer Interaction Networks")
            st.markdown("Analyze customer relationships, influence, or communities based on interaction data (e.g., reviews, social connections if available).")
            st.subheader("NGA 3: Centrality & Influence Measures")
            st.markdown("Identify key products, customers, or brands within a network based on their connectivity and influence.")
            st.subheader("NGA 4: Community Detection Algorithms")
            st.markdown("Uncover hidden communities or segments within product, customer, or brand networks.")
            st.subheader("NGA 5: Knowledge Graph Construction")
            st.markdown("Build a structured representation of your product catalog and related entities to enable complex queries and reasoning.")

        # Category 7: Optimization & Simulation (OSM)
        with st.expander("⚙️ Optimization & Simulation (OSM)"):
            st.subheader("OSM 1: Dynamic Pricing Optimization")
            st.markdown("Develop models to set optimal prices based on demand, competitor pricing, inventory levels, and other factors.")
            st.subheader("OSM 2: Inventory & Supply Chain Optimization")
            st.markdown("Use mathematical optimization techniques to manage inventory levels, reduce stockouts, and optimize supply chain logistics.")
            st.subheader("OSM 3: Marketing Mix Modeling & Budget Allocation")
            st.markdown("Optimize marketing spend across different channels to maximize ROI using econometric models.")
            st.subheader("OSM 4: Monte Carlo Simulation")
            st.markdown("Model uncertainty and simulate various scenarios for sales forecasts, profit projections, or risk assessment.")
            st.subheader("OSM 5: A/B Test Design & Power Analysis")
            st.markdown("Optimize the design of A/B tests to ensure sufficient statistical power and reliable results with minimal resources.")

        # Category 8: Advanced Visualization & Interaction (AVI)
        with st.expander("📊 Advanced Visualization & Interaction (AVI)"):
            st.subheader("AVI 1: Interactive Dashboards (Plotly, Bokeh)")
            st.markdown("Create highly interactive dashboards with features like drill-downs, filtering, and linked charts for deeper data exploration.")
            st.subheader("AVI 2: Geospatial Data Visualization (Folium, Kepler.gl)")
            st.markdown("Map sales data, customer locations, or store distributions on interactive maps, potentially with layers and heatmaps.")
            st.subheader("AVI 3: High-Dimensional Data Visualization")
            st.markdown("Use techniques like parallel coordinates plots, radar charts, or 3D scatter plots to visualize data with many features.")
            st.subheader("AVI 4: Sankey Diagrams & Sunburst Charts")
            st.markdown("Visualize flows (e.g., customer journeys, conversion funnels) or hierarchical data structures effectively.")
            st.subheader("AVI 5: Automated Insight Highlighting")
            st.markdown("Develop systems that automatically identify and visually highlight significant patterns or anomalies in dashboards.")

        # Category 9: Causal Inference & Experimentation (CIE)
        with st.expander("⚖️ Causal Inference & Experimentation (CIE)"):
            st.subheader("CIE 1: Propensity Score Matching (PSM)")
            st.markdown("Estimate the causal effect of a treatment or intervention in observational data by creating comparable groups.")
            st.subheader("CIE 2: Difference-in-Differences (DiD)")
            st.markdown("Analyze the impact of an intervention by comparing changes over time between a treatment group and a control group.")
            st.subheader("CIE 3: Uplift Modeling")
            st.markdown("Identify which customers are most likely to respond positively to a marketing intervention (persuadables).")
            st.subheader("CIE 4: Interrupted Time Series (ITS) Analysis")
            st.markdown("Assess the impact of an event or policy by analyzing changes in a time series before and after the intervention.")
            st.subheader("CIE 5: Instrumental Variables (IV) Regression")
            st.markdown("Address endogeneity and omitted variable bias to estimate causal relationships in observational data.")

        # Category 10: AI-Powered Strategic Insights (AISI)
        with st.expander("💡 AI-Powered Strategic Insights (AISI) - Advanced LLM Use"):
            st.subheader("AISI 1: Automated Market Trend & Competitor Analysis Reports")
            st.markdown("Leverage LLMs to synthesize information from product data, sales figures, and external sources to generate comprehensive reports.")
            st.subheader("AISI 2: Dynamic Persona Generation & Evolution")
            st.markdown("Use LLMs to create rich customer personas from segmentation data and update them as customer behavior changes.")
            st.subheader("AISI 3: Automated SWOT Analysis from Data")
            st.markdown("Prompt LLMs to perform a Strengths, Weaknesses, Opportunities, Threats analysis based on internal data and market context.")
            st.subheader("AISI 4: Predictive Narrative Generation for Business Reviews")
            st.markdown("Generate human-like narratives explaining key business performance indicators, trends, and forecasts for executive summaries.")
            st.subheader("AISI 5: Scenario Planning & Consequence Modeling with LLMs")
            st.markdown("Use LLMs to explore potential outcomes of strategic decisions by simulating different scenarios and their likely impacts based on available data and defined rules.")

except FileNotFoundError:
    st.error("🚨 Error: `FashionDataset.csv` not found. Please make sure the file is in the same directory as `app.py`.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("🚨 Error: `FashionDataset.csv` is empty. Please provide a valid CSV file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()
