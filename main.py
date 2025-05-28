from src.data.data_cleaning import pipeline as cleaning_pipeline
from src.data.preprocessing import pipeline as preprocessing_pipeline
from src.pipeline import run_model_pipeline
import matplotlib
matplotlib.use('Agg')


def main():
    cleaning_pipeline()
    preprocessing_pipeline()
    run_model_pipeline()


if __name__ == "__main__":
    main()
