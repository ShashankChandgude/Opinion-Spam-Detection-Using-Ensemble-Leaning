# Opinion-Spam-Detection-Using-Ensemble-Leaning

## Introduction

In today’s digital marketplace, customer reviews significantly influence business reputation. However, fake reviews can distort consumer perceptions and lead to poor decision making. This project addresses the challenge of detecting deceptive reviews by leveraging ensemble learning methods. While traditional methods such as SVMs have been popular, modern ensemble techniques—specifically bagging and stacking—have shown improved performance by combining the strengths of multiple classifiers.

This project uses a variety of base classifiers (including Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors, Multinomial Naive Bayes, Support Vector Machine, and Multilayer Perceptron) and integrates them into ensemble models. In our updated implementation:

- **Bagging** is applied to each base classifier to reduce variance.
- **Stacking** combines all optimized base models using a meta-estimator (defaulting to Logistic Regression) to produce a robust final prediction.

Every major step is logged to `output/log.txt` using a dedicated logger module, and all visualizations (such as cross-validation plots and error curves) are saved in the `output/model` directory.

## Project Structure

The project is organized into distinct modules that adhere to SOLID principles:

- **src/data_cleaning.py:**  
  Contains the pipeline for cleaning raw data and performing exploratory analysis. The cleaned data is saved to `data/processed/updated_data.csv`.

- **src/preprocessing.py:**  
  Performs text preprocessing steps (e.g., removal of special characters, stopwords, punctuation, and stemming) on the cleaned data.

- **src/vectorization.py:**  
  Loads the processed data, ensures there are no missing values in the text, splits it into training and test sets, and vectorizes the text using either CountVectorizer or TF-IDF.

- **src/cross_validation.py:**  
  Implements K-fold cross-validation on the training set and provides functions for plotting performance metrics (accuracy, precision, recall, F1 score).

- **src/hyperparameter_optimization.py:**  
  Contains functions to perform randomized hyperparameter search with K-fold cross-validation. It also includes hyperparameter configuration dictionaries (HYPERPARAMS) for each classifier.

- **src/model_training.py:**  
  Defines the base classifiers and provides two separate functions for ensemble training:  
  - `train_bagging_ensemble`: Trains a bagging ensemble for each base classifier.
  - `train_stacking_ensemble`: Creates a stacking ensemble by combining all optimized base models.  
  The `train_all_ensembles` function trains bagging ensembles for all classifiers and a single stacking ensemble.

- **src/evaluation_metrics.py:**  
  Contains functions for calculating evaluation metrics (accuracy, precision, recall, F1 score) for individual models and sets of models.

- **src/evaluation_visualization.py:**  
  Combines plotting and reporting functionality. It includes functions for plotting error curves and converting evaluation results into pandas DataFrames.

- **src/compute_errors.py:**  
  Provides a function to compute train and test errors for both optimized models and bagging ensembles.

- **src/utils.py:**  
  Contains common imports, utility functions (e.g., `get_project_root()`), and logging configuration.

- **main.py:**  
  Orchestrates the entire pipeline:
  - Runs the data cleaning and preprocessing pipelines.
  - Loads and vectorizes the processed data.
  - Performs cross-validation.
  - Optimizes hyperparameters.
  - Trains bagging ensembles for each classifier and a stacking ensemble that combines all optimized models.
  - Evaluates each model and ensemble.
  - Computes and plots error curves.
  - Logs every step and saves all visualizations to `output/model`.

## Logging and Visualizations

- **Logging:**  
  Every major step and output is logged to `output/log.txt` via the `logger_helper` module. This central logging approach ensures that the pipeline execution is traceable and debuggable.

- **Visualizations:**  
  All generated visual outputs (e.g., cross-validation results and error curves) are automatically saved in the `output/model` directory.

## How to Run

1. **Install Dependencies:**  
   Ensure that all required packages are installed (refer to `requirements.txt` for details).

2. **Execute the Pipeline:**  
   Run the main pipeline with:
   ```bash
   python main.py
