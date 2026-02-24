import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📊 AI Sales Forecasting Dashboard")

# 📁 File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv("sales.csv", encoding='latin1')

    # Clean columns
    df.columns = df.columns.str.strip()

    st.subheader("Raw Data")
    st.write(df.head())

    # Convert date
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Filters
    col1, col2 = st.columns(2)

    if 'Region' in df.columns:
        region = col1.selectbox("Select Region", df['Region'].unique())
        df = df[df['Region'] == region]

    if 'Category' in df.columns:
        category = col2.selectbox("Select Category", df['Category'].unique())
        df = df[df['Category'] == category]

    # Group by date
    df = df.groupby('Order Date')['Sales'].sum().reset_index()

    # Feature Engineering
    df['day'] = df['Order Date'].dt.day
    df['month'] = df['Order Date'].dt.month
    df['year'] = df['Order Date'].dt.year

    # Train/Test Split
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

    # 📊 Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    st.subheader("📈 Model Performance")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    # Forecast days input
    days = st.slider("Select number of days to forecast", 1, 30, 7)

    future_dates = pd.date_range(df['Order Date'].max(), periods=days+1)[1:]

    future_df = pd.DataFrame({'Order Date': future_dates})
    future_df['day'] = future_df['Order Date'].dt.day
    future_df['month'] = future_df['Order Date'].dt.month
    future_df['year'] = future_df['Order Date'].dt.year

    future_df['Predicted Sales'] = model.predict(future_df[['day','month','year']])

    # Show forecast
    st.subheader("📊 Future Forecast")
    st.write(future_df)

    # 📥 Download button
    csv = future_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

    # 📉 Charts
    st.subheader("📉 Sales Trend")

    st.line_chart(df.set_index('Order Date')['Sales'])
    st.line_chart(future_df.set_index('Order Date')['Predicted Sales'])