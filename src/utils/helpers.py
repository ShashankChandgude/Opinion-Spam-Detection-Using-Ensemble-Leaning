import os
import warnings
from functools import lru_cache
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
from src.utils.logging import logging

sns.set_theme(style="darkgrid")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning)

for resource in ['punkt', 'stopwords', 'punkt_tab']:
    nltk.download(resource, quiet=True)

@lru_cache(maxsize=1)
def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_verified_purchase_distribution(data: pd.DataFrame, out_folder: str, filename: str) -> None:
    vp_counts = data['verified_purchase'].value_counts()
    plt.figure(figsize=(4, 4))
    plt.pie(vp_counts.values, labels=vp_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Verified Purchase Distribution')
    filepath = os.path.join(out_folder, filename)
    plt.savefig(filepath)
    logging.info("Saved verified purchase distribution plot: %s", filepath)
    plt.close()

def plot_review_length_comparison(data: pd.DataFrame, out_folder: str, filename: str) -> None:
    subset = data[['verified_purchase', 'review_text']]
    avg_lengths = subset.groupby('verified_purchase')['review_text'].apply(lambda x: x.str.len().mean())
    
    plot_data = pd.DataFrame({
        'verified_purchase': ['True', 'False'],
        'avg_length': [avg_lengths.get(True, 0), avg_lengths.get(False, 0)]
    })
    
    plt.figure()
    sns.barplot(data=plot_data, x='avg_length', y='verified_purchase')
    plt.xlabel("Average Review Length")
    plt.ylabel("Verified Purchase")
    plt.title("Review Length Comparison")
    filepath = os.path.join(out_folder, filename)
    plt.savefig(filepath)
    logging.info("Saved review length comparison plot: %s", filepath)
    plt.close()

__all__ = [
    'os', 'warnings', 'logging', 'nltk', 'sns', 'pd', 'plt', 're', 'string',
    'get_project_root', 'WordCloud', 'TextBlob', 'stopwords',
    'word_tokenize', 'PorterStemmer', 'plot_verified_purchase_distribution',
    'plot_review_length_comparison'
]
