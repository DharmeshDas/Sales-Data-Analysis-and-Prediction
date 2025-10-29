import streamlit as st
import pandas as pd
import plotly.express as px
import chardet

# ---- Safe CSV Reader (no warnings) ----
def read_csv_safely(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except UnicodeDecodeError:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                detected_encoding = result['encoding']
                df = pd.read_csv(file_path, encoding=detected_encoding)
    return df

# ---- Streamlit UI ----
st.set_page_config(page_title="Sales Data Dashboard", layout="wide")
st.title("Sales Data Analysis Dashboard")

# Load dataset safely (silently)
df = read_csv_safely("data/superstore.csv")

# Clean date
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# Sidebar filter
region = st.sidebar.selectbox("Select Region", sorted(df['Region'].dropna().unique()))

# Filter data
filtered_df = df[df['Region'] == region]

# Monthly Sales Trend
monthly_sales = filtered_df.groupby('Month')['Sales'].sum().reset_index()
fig1 = px.line(monthly_sales, x='Month', y='Sales',
title=f"Monthly Sales Trend ({region})", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# Category Sales
cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
fig2 = px.bar(cat_sales, x='Category', y='Sales',
title=f"Sales by Category ({region})")
st.plotly_chart(fig2, use_container_width=True)

# Profit by Sub-Category
if 'Profit' in df.columns:
    sub_profit = filtered_df.groupby('Sub-Category')['Profit'].sum().reset_index()
    fig3 = px.bar(sub_profit, x='Sub-Category', y='Profit',
    title=f"Profit by Sub-Category ({region})")
    st.plotly_chart(fig3, use_container_width=True)

# Clean success message (optional)
st.caption("Dashboard loaded successfully.")
