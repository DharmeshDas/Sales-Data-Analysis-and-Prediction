import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir="reports/figures"):
    """
    Performs Exploratory Data Analysis and saves the generated figures to disk.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Month', 'Sales', 'Category', etc.
        output_dir (str): Directory where the plots will be saved.
    """
    print("Performing Exploratory Data Analysis...")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sales by category
    if 'Category' in df.columns and 'Sales' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Category', y='Sales', data=df, estimator=sum, palette='viridis')
        plt.title("Total Sales by Category")
        plt.tight_layout() # Ensures labels are not cut off
        plt.savefig(os.path.join(output_dir, "eda_sales_by_category.png"))
        plt.close() # Close the figure to free up memory
        print(f"Saved: {output_dir}/eda_sales_by_category.png")

    # 2. Monthly sales trend
    if 'Month' in df.columns and 'Sales' in df.columns:
        monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='Month', y='Sales', data=monthly_sales, marker='o')
        plt.title("Monthly Sales Trend")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_monthly_sales_trend.png"))
        plt.close()
        print(f"Saved: {output_dir}/eda_monthly_sales_trend.png")

    # 3. Regional performance
    if 'Region' in df.columns and 'Profit' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Region', y='Profit', data=df, estimator=sum, palette='plasma')
        plt.title("Profit by Region")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_profit_by_region.png"))
        plt.close()
        print(f"Saved: {output_dir}/eda_profit_by_region.png")

    print("EDA completed and figures saved successfully.")

# Note: The plt.show() calls were removed and replaced with plt.savefig()