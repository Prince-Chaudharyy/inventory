import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Inventory Management Dashboard", layout="wide")

st.title("📦 Demand-Based Inventory Management System")
st.caption("Organization: Flavi Dairy Solutions, Ahmedabad")

uploaded_file = st.file_uploader("Upload Inventory CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded from: '{uploaded_file.name}'")

    st.subheader("🗃️ Raw Inventory Data")
    st.dataframe(df.head(10))

    # Demand Forecasting
    st.subheader("📈 Demand Forecasting (7-Day Outlook)")
    usage_data = df[["Quantity_Used (Liters/Kg/Units)"]].reset_index()
    usage_data.columns = ["Day", "Quantity"]

    # Linear Regression Model
    model = LinearRegression()
    X = usage_data["Day"].values.reshape(-1, 1)
    y = usage_data["Quantity"].values
    model.fit(X, y)

    future_days = np.array(range(len(usage_data), len(usage_data)+7)).reshape(-1, 1)
    predictions = model.predict(future_days)

    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Predicted Quantity": predictions.round(2)
    })

    st.metric("📊 Forecast Accuracy (MAE)", round(np.mean(abs(model.predict(X) - y)), 2))
    st.dataframe(forecast_df)

    # Line Plot
    fig1, ax1 = plt.subplots()
    ax1.plot(usage_data["Day"], usage_data["Quantity"], marker='o', label="Actual")
    ax1.plot(future_days.flatten(), predictions, marker='x', linestyle='--', label="Forecast")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Quantity Used")
    ax1.legend()
    st.pyplot(fig1)

    # Alerts Section
    st.subheader("🚨 Low Stock Alerts")
    threshold = st.slider("Set Stock Threshold", min_value=0, max_value=1000, value=100, step=10)
    low_stock_items = df[df["Stock_On_Hand"] < threshold]
    if not low_stock_items.empty:
        st.warning("⚠️ Items below stock threshold:")
        st.dataframe(low_stock_items[["Material_Name", "Stock_On_Hand"]])
    else:
        st.success("✅ All inventory levels are above the threshold.")

    # Vendor Performance
    st.subheader("📦 Vendor Performance Overview")
    avg_delivery = df.groupby("Vendor_Name")["Delivery_Lead_Time (Days)"].mean().sort_values()
    st.bar_chart(avg_delivery)

else:
    st.warning("📁 Please upload a CSV file to begin.")
