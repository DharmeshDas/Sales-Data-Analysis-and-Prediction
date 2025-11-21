import sys
import os
from io import BytesIO
from datetime import timedelta

# Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Optional: if you have utils
try:
    from src.utils import load_config
except Exception:
    load_config = None

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import chardet
from sklearn.linear_model import LinearRegression


prophet_available = True
try:
    from prophet import Prophet
except Exception:
    try:
        # older name
        from fbprophet import Prophet  # type: ignore
    except Exception:
        prophet_available = False

# Try to import ReportLab for PDF export
reportlab_available = True
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
except Exception:
    reportlab_available = False

# -----------------------
# Utilities / Data Load
# -----------------------
@st.cache_data
def read_csv_safely(file_path: str) -> pd.DataFrame:
    """Read CSV with encoding fallback and small detection fallback."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='latin1')
        except UnicodeDecodeError:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(100000))
                detected_encoding = result.get('encoding', 'utf-8')
            return pd.read_csv(file_path, encoding=detected_encoding)

# -----------------------
# Config / Load Data
# -----------------------
st.set_page_config(page_title="Sales Data Analysis Dashboard ", layout="wide")

st.title(" Sales Data Analysis ")

# Load dataset (ensure path is correct relative to project root)
DATA_PATH = os.path.join(project_root, "data", "superstore.csv")
try:
    df = read_csv_safely(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Basic preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df = df.dropna(subset=['Order Date'])
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year
df['Month_Name'] = df['Order Date'].dt.strftime('%b')
months_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=months_order, ordered=True)

# Ensure numeric columns
for col in ['Sales','Profit','Quantity','Discount']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# -----------------------
# Sidebar: Filters
# -----------------------
st.sidebar.header(" Filters & Options")
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()

date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) != 2:
    st.sidebar.error("Please select start and end date")
    st.stop()

start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
regions = st.sidebar.multiselect("Region(s)", sorted(df['Region'].dropna().unique()), default=sorted(df['Region'].dropna().unique()))
categories = st.sidebar.multiselect("Category(ies)", sorted(df['Category'].dropna().unique()), default=sorted(df['Category'].dropna().unique()))
compare_by = st.sidebar.selectbox("Comparison dimension", options=["Region","Category"])

# Apply filters
filtered_df = df[
    (df['Order Date'] >= start_date) &
    (df['Order Date'] < end_date) &
    (df['Region'].isin(regions)) &
    (df['Category'].isin(categories))
].copy()

if filtered_df.empty:
    st.warning("No data available for selected filters. Adjust filters to see visualizations.")
    st.stop()

# -----------------------
# Top KPIs (global for the applied filters)
# -----------------------
st.markdown("##  Key Performance Indicators (Filtered)")
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = filtered_df['Order ID'].nunique() if 'Order ID' in filtered_df.columns else len(filtered_df)
avg_profit_margin = (total_profit / total_sales * 100) if total_sales != 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("🛒 Total Sales", f"${total_sales:,.0f}")
k2.metric("💰 Total Profit", f"${total_profit:,.0f}")
k3.metric("📦 Total Orders", f"{total_orders}")
k4.metric("📈 Avg Profit Margin", f"{avg_profit_margin:.2f}%")

# -----------------------
# Comparison Tool
# -----------------------
st.markdown("##  Sales Comparison by {compare_by}")
comp_df = filtered_df.groupby(compare_by)[['Sales', 'Profit']].sum().reset_index() # type: ignore
fig_comp = px.bar(comp_df.sort_values('Sales', ascending=False), x=compare_by, y='Sales',
                  hover_data=['Profit'], )
st.plotly_chart(fig_comp, use_container_width=True)
st.write(f" **Insight:** Top {compare_by} by sales: **{comp_df.loc[comp_df['Sales'].idxmax(), compare_by]}**.")

st.markdown("---")

# -----------------------
# Global Charts (both old and improved) with summaries under each chart
# -----------------------
st.markdown("##  Global Charts")

left_col, right_col = st.columns(2)

# Global Yearly Sales Trend
profit_category = df.groupby(['Category', 'Year'])['Profit'].sum().reset_index()
fig_profit_category = px.bar(
    profit_category, x='Year', y='Profit', color='Category',
    title=" Profit by Category Over Years", barmode='stack'
)
left_col.plotly_chart(fig_profit_category, use_container_width=True)
top_cat_profit = profit_category.groupby('Category')['Profit'].sum().idxmax()
left_col.write(f" **Insight:** The most profitable category overall is **{top_cat_profit}**, "f"suggesting strong customer demand and profit margin in this segment.")

# OLD: Profit by Region (original style)
region_profit_orig = filtered_df.groupby('Region')['Profit'].sum().reset_index()
fig_region_orig = px.bar(region_profit_orig, x='Region', y='Profit', title="Profit by Region (Original)", color='Profit', color_continuous_scale='rdbu')
right_col.plotly_chart(fig_region_orig, use_container_width=True)
right_col.write(f" **Insight:** Most profitable region: **{region_profit_orig.loc[region_profit_orig['Profit'].idxmax(),'Region']}** with ${region_profit_orig['Profit'].max():,.0f} profit.")

st.markdown("---")

# NEW: Monthly Sales Trend (Month names) and Sales by Category
ncol1, ncol2 = st.columns(2)

monthly_named = filtered_df.groupby('Month_Name')['Sales'].sum().reset_index()
monthly_named = monthly_named.sort_values('Month_Name')
fig_month_named = px.line(monthly_named, x='Month_Name', y='Sales', title="Monthly Sales Trend (Named Months)", markers=True)
ncol1.plotly_chart(fig_month_named, use_container_width=True)
ncol1.write(f" **Insight:** Peak month: **{monthly_named.loc[monthly_named['Sales'].idxmax(),'Month_Name']}** with ${monthly_named['Sales'].max():,.0f} sales. Consider promotions around this period.")

category_sales_global = filtered_df.groupby('Category')['Sales'].sum().reset_index()
fig_cat_sales = px.bar(category_sales_global.sort_values('Sales', ascending=False), x='Category', y='Sales', title="Sales by Category (Global)", color='Category')
ncol2.plotly_chart(fig_cat_sales, use_container_width=True)
ncol2.write(f" **Insight:** Top category by sales: **{category_sales_global.loc[category_sales_global['Sales'].idxmax(),'Category']}** with ${category_sales_global['Sales'].max():,.0f} in sales.")

st.markdown("---")

# -----------------------
# Regional / Detailed section (with region-level KPIs)
# -----------------------
st.markdown("## Detailed Analysis (Select a Primary Region)")

primary_region = st.selectbox("Primary Region for detail view", options=sorted(filtered_df['Region'].unique()))
region_df = filtered_df[filtered_df['Region'] == primary_region].copy()

# region KPIs
r_sales = region_df['Sales'].sum()
r_profit = region_df['Profit'].sum()
r_orders = region_df['Order ID'].nunique() if 'Order ID' in region_df.columns else len(region_df)
r_margin = (r_profit / r_sales * 100) if r_sales != 0 else 0

rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric(f"🛒 Total Sales ({primary_region})", f"${r_sales:,.0f}")
rc2.metric(f"💰 Total Profit ({primary_region})", f"${r_profit:,.0f}")
rc3.metric(f"📦 Total Orders ({primary_region})", f"{r_orders}")
rc4.metric(f"📈 Avg Profit Margin ({primary_region})", f"{r_margin:.2f}%")

# Regional charts with insights
rcol1, rcol2 = st.columns(2)

cat_sales_region = region_df.groupby('Category')['Sales'].sum().reset_index()
fig_cat_region = px.bar(cat_sales_region.sort_values('Sales', ascending=False), x='Category', y='Sales', title=f"Sales by Category ({primary_region})", color='Category')
rcol1.plotly_chart(fig_cat_region, use_container_width=True)
rcol1.write(f" **Insight:** In {primary_region}, top category is **{cat_sales_region.loc[cat_sales_region['Sales'].idxmax(),'Category']}**.")

sub_profit = region_df.groupby('Sub-Category')['Profit'].sum().reset_index()
fig_sub_profit = px.bar(sub_profit.sort_values('Profit', ascending=False), x='Sub-Category', y='Profit', title=f"Profit by Sub-Category ({primary_region})", color='Profit', color_continuous_scale='rdbu')
rcol2.plotly_chart(fig_sub_profit, use_container_width=True)
rcol2.write(f" **Insight:** In {primary_region}, most profitable sub-category is **{sub_profit.loc[sub_profit['Profit'].idxmax(),'Sub-Category']}**.")

st.markdown("---")

# -----------------------
# ML-Based Trend Analysis & Forecast
# -----------------------
st.markdown("## Trend Analysis & Forecasting")

# Aggregated monthly series for trend & forecasting
monthly_ts = filtered_df.set_index('Order Date').resample('M')['Sales'].sum().reset_index()
monthly_ts['ds'] = monthly_ts['Order Date']
monthly_ts['y'] = monthly_ts['Sales']

# Use Prophet if available, otherwise LinearRegression fallback
forecast_days = st.sidebar.number_input("Forecast months (Prophet)/periods (LR)", min_value=1, max_value=12, value=3)

if prophet_available:
    try:
        m = Prophet()
        m.fit(monthly_ts[['ds','y']])
        future = m.make_future_dataframe(periods=forecast_days, freq='M')
        forecast = m.predict(future)
        fig_forecast = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast (yhat)")
        st.plotly_chart(fig_forecast, use_container_width=True)
        # Insights
        forecast_future = forecast[forecast['ds'] > monthly_ts['ds'].max()]
        predicted_sum = forecast_future['yhat'].sum()
        st.write(f" **Forecast Insight (Prophet):** Predicted sales for next {forecast_days} months ≈ **${predicted_sum:,.0f}**.")
    except Exception as e:
        st.error(f"Prophet forecasting failed: {e}")
        prophet_available = False

if not prophet_available:
    # Linear Regression fallback on monthly_ts
    lr_df = monthly_ts.copy().reset_index(drop=True)
    lr_df = lr_df.sort_values('ds')
    X = np.arange(len(lr_df)).reshape(-1, 1)
    y = lr_df['y'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    # predict next forecast_days months
    future_idx = np.arange(len(lr_df), len(lr_df) + forecast_days).reshape(-1, 1)
    preds = model.predict(future_idx).flatten()
    # Build display df
    future_dates = pd.date_range(start=lr_df['ds'].max() + pd.offsets.MonthBegin(1), periods=forecast_days, freq='M')
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': preds})
    fig_lin = px.line(pd.concat([lr_df[['ds','y']].rename(columns={'y':'yhat'}), forecast_df]), x='ds', y='yhat', title="Linear Regression Forecast")
    st.plotly_chart(fig_lin, use_container_width=True)
    st.write(f" **Forecast Insight (Linear Reg):** Predicted sales for next {forecast_days} months ≈ **${forecast_df['yhat'].sum():,.0f}**.")

st.markdown("---")

# -----------------------
# Decision Recommendations (simple rule-based)
# -----------------------
st.markdown("##  Decision Recommendations")
mean_sales = monthly_ts['Sales'].mean()
last_month = monthly_ts.iloc[-1]['Sales'] if not monthly_ts.empty else 0

if last_month > mean_sales:
    st.success(" Sales are above average this period. Recommendation: Increase stock and marketing for top categories.")
else:
    st.warning(" Sales are below average. Recommendation: Run promotions targeting top-performing categories and regions.")

# High-level automated recommendation example
top_cat_global = category_sales_global.loc[category_sales_global['Sales'].idxmax(), 'Category']
st.write(f" **Actionable:** Consider promoting **{top_cat_global}**, the top-selling category in the selected filters.")

st.markdown("---")

# -----------------------
# Export / Download (CSV/Excel & PDF)
# -----------------------
st.markdown("## 📤 Export & Reports")

# CSV / Excel download
csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Data (CSV)", data=csv_bytes, file_name="filtered_sales.csv", mime="text/csv")

# Excel (xlsx) using BytesIO
def to_excel_bytes(df_input: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_input.to_excel(writer, index=False, sheet_name='Sales')
    return output.getvalue()

xlsx_data = to_excel_bytes(filtered_df)
st.download_button("📥 Download Filtered Data (Excel)", data=xlsx_data, file_name="filtered_sales.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# PDF summary generation
def create_pdf_summary_bytes():
    output = BytesIO()
    if reportlab_available:
        c = canvas.Canvas(output, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 40, "Sales Report Summary")
        c.setFont("Helvetica", 11)
        lines = [
            f"Date Range: {start_date.date()} to { (end_date - pd.Timedelta(days=1)).date()}",
            f"Filtered Regions: {', '.join(regions)}",
            f"Filtered Categories: {', '.join(categories)}",
            f"Total Sales: ${total_sales:,.0f}",
            f"Total Profit: ${total_profit:,.0f}",
            f"Top Category: {top_cat_global}",
        ]
        y = height - 80
        for line in lines:
            c.drawString(40, y, line)
            y -= 18
        c.showPage()
        c.save()
        output.seek(0)
        return output.read()
    else:
        # fallback: simple text file as PDF replacement
        text = (
            "Sales Report Summary\n\n" +
            f"Date Range: {start_date.date()} to { (end_date - pd.Timedelta(days=1)).date()}\n" +
            f"Regions: {', '.join(regions)}\n" +
            f"Categories: {', '.join(categories)}\n" +
            f"Total Sales: ${total_sales:,.0f}\n" +
            f"Total Profit: ${total_profit:,.0f}\n" +
            f"Top Category: {top_cat_global}\n"
        )
        output.write(text.encode('utf-8'))
        output.seek(0)
        return output.read()

pdf_bytes = create_pdf_summary_bytes()
st.download_button("📥 Download Summary Report (PDF)", data=pdf_bytes, file_name="sales_summary.pdf", mime="application/pdf")

st.markdown("---")

# -----------------------
# Executive Summary (auto-generated)
# -----------------------
st.markdown("## Executive Summary")
best_region_overall = comp_df.loc[comp_df['Sales'].idxmax(), compare_by]
most_profitable_category = category_sales_global.loc[category_sales_global['Sales'].idxmax(), 'Category']
st.write(f"""
-  **Top {compare_by} by Sales:** **{best_region_overall}**
-  **Top Category (by sales):** **{most_profitable_category}**
-  **Forecast (next {forecast_days} months):** see Forecast section above.
""")
