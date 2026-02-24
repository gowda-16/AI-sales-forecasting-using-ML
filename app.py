import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page config
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")

# Custom CSS (🔥 UI Enhancement)
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1c1f26; padding: 10px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚀 AI Sales Forecasting Dashboard")
st.markdown("### 📊 Predict future sales with Machine Learning")

# Sidebar
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    df.columns = df.columns.str.strip()

    # Date processing
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('Order Date')

    # Filters
    if 'Region' in df.columns:
        region = st.sidebar.selectbox("🌍 Region", df['Region'].unique())
        df = df[df['Region'] == region]

    if 'Category' in df.columns:
        category = st.sidebar.selectbox("📦 Category", df['Category'].unique())
        df = df[df['Category'] == category]

    # Group data
    df = df.groupby('Order Date')['Sales'].sum().reset_index()

    # Features
    df['day'] = df['Order Date'].dt.day
    df['month'] = df['Order Date'].dt.month
    df['year'] = df['Order Date'].dt.year

    # Split
    split = int(len(df) * 0.8)
    train = df[:split]
    test = df[split:]

    X_train = train[['day','month','year']]
    y_train = train['Sales']

    X_test = test[['day','month','year']]
    y_test = test['Sales']

    # Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # KPI Section
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("📉 MAE", f"{mae:.2f}")
    col2.metric("📊 RMSE", f"{rmse:.2f}")
    col3.metric("📦 Total Sales", f"{df['Sales'].sum():,.0f}")

    # Forecast days
    days = st.sidebar.slider("📅 Forecast Days", 1, 30, 7)

    future_dates = pd.date_range(df['Order Date'].max(), periods=days+1)[1:]

    future_df = pd.DataFrame({'Order Date': future_dates})
    future_df['day'] = future_df['Order Date'].dt.day
    future_df['month'] = future_df['Order Date'].dt.month
    future_df['year'] = future_df['Order Date'].dt.year

    future_df['Predicted Sales'] = model.predict(future_df[['day','month','year']])

    # Charts
    st.subheader("📈 Sales Trend")

    col4, col5 = st.columns(2)

    col4.line_chart(df.set_index('Order Date')['Sales'])
    col5.line_chart(future_df.set_index('Order Date')['Predicted Sales'])

    # Forecast Table
    st.subheader("📊 Future Predictions")
    st.dataframe(future_df)

    # Download
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast", csv, "forecast.csv")

else:
    st.warning("👈 Upload a dataset to start")