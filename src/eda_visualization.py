import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("Performing Exploratory Data Analysis...")

    # 1. Sales by category
    if 'Category' in df.columns and 'Sales' in df.columns:
        plt.figure(figsize=(8,5))
        sns.barplot(x='Category', y='Sales', data=df, estimator=sum)
        plt.title("Total Sales by Category")
        plt.show()

    # 2. Monthly sales trend
    if 'Month' in df.columns and 'Sales' in df.columns:
        monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
        plt.figure(figsize=(10,5))
        sns.lineplot(x='Month', y='Sales', data=monthly_sales, marker='o')
        plt.title("Monthly Sales Trend")
        plt.show()

    # 3. Regional performance
    if 'Region' in df.columns and 'Profit' in df.columns:
        plt.figure(figsize=(8,5))
        sns.barplot(x='Region', y='Profit', data=df, estimator=sum)
        plt.title("Profit by Region")
        plt.show()

    print("EDA completed successfully.")
