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

# Imports for potential advanced analytics tools (some may be placeholders)
import scipy.stats as stats
import statsmodels.api as sm
# import networkx as nx # For graph analysis
# import plotly.express as px # For advanced interactive visualizations
# import shap # For model interpretability
# from gensim.models import LdaModel # For topic modeling
# # import tensorflow as tf # For deep learning (conceptual)
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
            st.subheader("ASM 1: Advanced Hypothesis Testing")
            st.markdown("Utilize tests like Chi-Squared for categorical associations, ANOVA/MANOVA for comparing multiple group means, or non-parametric tests for non-normal data.")
            st.subheader("ASM 2: Logistic Regression")
            st.markdown("Model binary outcomes (e.g., predict high/low discount, purchase likelihood) based on various product or customer features.")
            st.subheader("ASM 3: Time Series Decomposition & Analysis")
            st.markdown("Break down time series data (e.g., sales over time) into trend, seasonality, and residuals. Apply models like ARIMA, SARIMA, or Prophet for forecasting.")
            st.subheader("ASM 4: Survival Analysis")
            st.markdown("Analyze the time until an event occurs, such as product lifecycle duration, customer churn, or time-to-next-purchase.")
            st.subheader("ASM 5: Bayesian A/B Testing")
            st.markdown("Apply Bayesian methods to A/B testing for more nuanced insights, including probabilities of one variant being better than another.")

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
