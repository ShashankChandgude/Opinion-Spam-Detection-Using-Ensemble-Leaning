import os
import warnings
import nltk
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from src.utils.logging import logging, configure_logging

sns.set_theme(style="darkgrid")
warnings.simplefilter("ignore")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_verified_purchase_distribution(data: pd.DataFrame, out_folder: str, filename: str) -> None:
    plt.figure(figsize=(4, 4))
    vp_counts = data['verified_purchase'].value_counts()
    plt.pie(vp_counts.values, labels=vp_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Verified Purchase Distribution')
    filepath = os.path.join(out_folder, filename)
    plt.savefig(filepath)
    logging.info("Saved verified purchase distribution plot: %s", filepath)
    plt.close()

def plot_review_length_comparison(data: pd.DataFrame, out_folder: str, filename: str) -> None:
    subset = data[['verified_purchase', 'review_text']]
    true_reviews = subset[subset['verified_purchase'] == True]
    false_reviews = subset[subset['verified_purchase'] == False]
    avg_true = true_reviews['review_text'].apply(len).mean()
    avg_false = false_reviews['review_text'].apply(len).mean()
    plt.figure()
    sns.barplot(x=[avg_true, avg_false], y=["True", "False"])
    plt.xlabel("Average Review Length")
    plt.ylabel("Verified Purchase")
    plt.title("Review Length Comparison")
    filepath = os.path.join(out_folder, filename)
    plt.savefig(filepath)
    logging.info("Saved review length comparison plot: %s", filepath)
    plt.close()

__all__ = [
    'os', 'warnings', 'logging', 'nltk', 'sns', 'pd', 'plt', 're', 'string',
    'get_project_root', 'configure_logging', 'WordCloud', 'TextBlob', 'stopwords',
    'word_tokenize', 'PorterStemmer', 'plot_verified_purchase_distribution',
    'plot_review_length_comparison'
]
