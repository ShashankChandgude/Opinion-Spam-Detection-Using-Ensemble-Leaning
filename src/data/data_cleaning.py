#!/usr/bin/env python
# coding: utf-8

from src.utils.helpers import os, pd, plt, sns, get_project_root, plot_verified_purchase_distribution, plot_review_length_comparison
from src.data.data_io import load_csv_file, write_csv_file
from src.utils.logging import logging, configure_logging
import matplotlib
matplotlib.use('Agg')

sns.set_theme()

def explore_data(data: pd.DataFrame) -> None:
    logging.info("Rows: %d Columns: %d", data.shape[0], data.shape[1])
    data.info(buf=open(os.devnull, "w"))
    logging.info("Head:\n%s", data.head().to_string())
    logging.info("Numerical stats:\n%s", data.describe().to_string())
    logging.info("Categorical stats:\n%s", data.describe(include=object).to_string())

def fix_column_names(data: pd.DataFrame) -> pd.DataFrame:
    if 'ï»¿report_date' in data.columns:
        data.rename(columns={'ï»¿report_date': 'report_date'}, inplace=True)
    return data

def update_categories(data: pd.DataFrame) -> pd.DataFrame:
    data['category'] = data['category'].astype(object)
    data.loc[data.sub_category == "Ice Cream", "category"] = "Refreshment"
    data.loc[data.sub_category == "HHC", "sub_category"] = "Household Care"
    data.loc[data.sub_category == "Deos", "sub_category"] = "Deodorants & Fragrances"
    data.loc[data.sub_category == "Tea", "sub_category"] = "Tea and Soy & Fruit Beverages"
    data.loc[data.sub_category == "Hair Care", "sub_category"] = "Hair"
    return data

def remove_unneeded_columns(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        'matched_keywords', 'time_of_publication', 'manufacturers_response',
        'dimension4', 'dimension5', 'dimension6', 'is_competitor',
        'report_date', 'online_store', 'brand',
        'category', 'sub_category', 'market', 'upc', 'retailer_product_code',
        'review_hash_id', 'url', 'product_description', 'parent_review',
        'review_type', 'manufacturer', 'dimension1', 'dimension2', 'dimension3',
        'dimension7', 'dimension8'
    ]
    data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return data

def plot_helpful_review_counts(data: pd.DataFrame) -> None:
    plt.figure()
    sns.countplot(x='helpful_review_count', data=data).set_title("Helpful Review Counts")
    root = get_project_root()
    plots_dir = os.path.join(root, "output", "data_cleaning")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "helpful_review_counts.png")
    plt.savefig(path)
    logging.info("Saved plot: %s", path)
    plt.close()

def clean_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    explore_data(data)
    data = fix_column_names(data)
    data = update_categories(data)
    logging.info("Duplicates: %d", data.duplicated().sum())
    data = remove_unneeded_columns(data)
    return data

def pipeline():
    root = get_project_root()
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "log.txt")
    configure_logging(log_file)

    logging.info("Starting data cleaning process")
    raw_file = os.path.join(root, "data", "raw", "Amazon_review_data.csv")
    proc_file = os.path.join(root, "data", "processed", "updated_data.csv")
    raw_data = load_csv_file(raw_file)
    cleaned_data = clean_pipeline(raw_data)
    plot_helpful_review_counts(cleaned_data)
    plots_dir = os.path.join(root, "output", "data_cleaning")
    os.makedirs(plots_dir, exist_ok=True)
    plot_verified_purchase_distribution(cleaned_data, plots_dir, "verified_purchase_distribution.png")
    plot_review_length_comparison(cleaned_data, plots_dir, "review_length_vs_verified_purchase.png")
    write_csv_file(cleaned_data, proc_file)
    logging.info("Data cleaning process completed successfully.")

def explore_data(data: pd.DataFrame) -> None:
    logging.info("Data shape: %d rows × %d columns", *data.shape)

    # show column names and dtypes
    dtype_summary = data.dtypes.to_frame("dtype")
    logging.info("Columns & dtypes:\n%s", dtype_summary.to_string())

    # numeric summary: only mean, min, max
    num = data.select_dtypes(include="number")
    if not num.empty:
        stats = num.describe().loc[["mean", "min", "max"]]
        logging.info("Numeric summary (mean/min/max):\n%s", stats.to_string())

    # categorical summary: unique counts
    cat = data.select_dtypes(include="object")
    if not cat.empty:
        uniques = cat.nunique().to_frame("unique_vals")
        logging.info("Categorical unique counts:\n%s", uniques.to_string())

def fix_column_names(data: pd.DataFrame) -> pd.DataFrame:
    if 'ï»¿report_date' in data.columns:
        data.rename(columns={'ï»¿report_date': 'report_date'}, inplace=True)
    return data

def update_categories(data: pd.DataFrame) -> pd.DataFrame:
    data['category'] = data['category'].astype(object)
    data.loc[data.sub_category == "Ice Cream", "category"] = "Refreshment"
    data.loc[data.sub_category == "HHC", "sub_category"] = "Household Care"
    data.loc[data.sub_category == "Deos", "sub_category"] = "Deodorants & Fragrances"
    data.loc[data.sub_category == "Tea", "sub_category"] = "Tea and Soy & Fruit Beverages"
    data.loc[data.sub_category == "Hair Care", "sub_category"] = "Hair"
    return data

def remove_unneeded_columns(data: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        'matched_keywords', 'time_of_publication', 'manufacturers_response',
        'dimension4', 'dimension5', 'dimension6', 'is_competitor',
        'report_date', 'online_store', 'brand',
        'category', 'sub_category', 'market', 'upc', 'retailer_product_code',
        'review_hash_id', 'url', 'product_description', 'parent_review',
        'review_type', 'manufacturer', 'dimension1', 'dimension2', 'dimension3',
        'dimension7', 'dimension8'
    ]
    data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return data

def plot_helpful_review_counts(data: pd.DataFrame) -> None:
    plt.figure()
    sns.countplot(x='helpful_review_count', data=data).set_title("Helpful Review Counts")
    root = get_project_root()
    plots_dir = os.path.join(root, "output", "data_cleaning")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "helpful_review_counts.png")
    plt.savefig(path)
    logging.info("Saved plot: %s", path)
    plt.close()

def clean_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    explore_data(data)
    data = fix_column_names(data)
    data = update_categories(data)
    return remove_unneeded_columns(data)

def pipeline():
    root = get_project_root()
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "log.txt")
    configure_logging(log_file)

    logging.info("🔹 Starting data cleaning")

    raw_file = os.path.join(root, "data", "raw", "Amazon_review_data.csv")
    raw_data = load_csv_file(raw_file)

    cleaned_data = clean_pipeline(raw_data)
    dupes = raw_data.shape[0] - cleaned_data.shape[0]
    logging.info("➔ Duplicates dropped: %d", dupes)

    plot_helpful_review_counts(cleaned_data)
    plot_verified_purchase_distribution(cleaned_data, os.path.join(out_dir, "data_cleaning"), "verified_purchase_distribution.png")
    plot_review_length_comparison(cleaned_data, os.path.join(out_dir, "data_cleaning"), "review_length_vs_verified_purchase.png")
    write_csv_file(cleaned_data, os.path.join(root, "data", "processed", "updated_data.csv"))

    logging.info("✅ Data cleaning done")

if __name__ == "__main__":
    pipeline() # pragma: no cover
