import argparse
from src.data.data_cleaning import pipeline as cleaning_pipeline
from src.data.preprocessing import pipeline as preprocessing_pipeline
from src.pipeline import model_pipeline
import matplotlib
matplotlib.use("Agg") 

def main():
    cleaning_pipeline()
    preprocessing_pipeline()
    parser = argparse.ArgumentParser(description="Opinion-Spam Detection: full train/eval pipeline")
    parser.add_argument("--vectorizer",choices=["count","tfidf"],default="tfidf",help="which text‐vectorizer to use")
    parser.add_argument("--test-size",type=float,default=0.2,help="fraction of data to hold out for test")
    args = parser.parse_args()
    model_pipeline(vectorizer_type=args.vectorizer,test_size=args.test_size)

if __name__ == "__main__":
    main()
