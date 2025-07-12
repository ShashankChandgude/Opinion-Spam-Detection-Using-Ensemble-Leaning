# Opinion-Spam-Detection-Using-Ensemble-Leaning

## Introduction

In today's digital marketplace, customer reviews significantly influence business reputation. However, fake reviews can distort consumer perceptions and lead to poor decision making. This project addresses the challenge of detecting deceptive reviews by leveraging ensemble learning methods. While traditional methods such as SVMs have been popular, modern ensemble techniques—specifically bagging and stacking—have shown improved performance by combining the strengths of multiple classifiers.

This project uses a variety of base classifiers (including Logistic Regression, Decision Trees, Random Forest, K-Nearest Neighbors, Multinomial Naive Bayes, Support Vector Machine, and Multilayer Perceptron) and integrates them into ensemble models. In our updated implementation:

- **Bagging** is applied to each base classifier to reduce variance.
- **Stacking** combines all optimized base models using a meta-estimator (defaulting to Logistic Regression) to produce a robust final prediction.

Every major step is logged to `output/log.txt` using a dedicated logger module, and all visualizations (such as cross-validation plots and error curves) are saved in the `output/plots` directory.

## Project Structure

The project is organized into distinct modules that adhere to SOLID principles:

### Data Processing
- **src/data/data_cleaning.py:**  
  Contains the pipeline for cleaning raw data and performing exploratory analysis. The cleaned data is saved to `data/processed/cleaned_data.csv`.

- **src/data/preprocessing.py:**  
  Performs text preprocessing steps (e.g., removal of special characters, stopwords, punctuation, and stemming) on the cleaned data.

- **src/data/load_data.py:**  
  Handles data loading operations with proper error handling and validation.

- **src/data/data_io.py:**  
  Provides data input/output utilities and file operations.

### Feature Engineering
- **src/features/vectorization.py:**  
  Loads the processed data, ensures there are no missing values in the text, splits it into training and test sets, and vectorizes the text using either CountVectorizer or TF-IDF.

### Model Training
- **src/training/cross_validation.py:**  
  Implements K-fold cross-validation on the training set and provides functions for plotting performance metrics (accuracy, precision, recall, F1 score).

- **src/training/hyperparameter_optimization.py:**  
  Contains functions to perform randomized hyperparameter search with K-fold cross-validation. It also includes hyperparameter configuration dictionaries for each classifier.

- **src/training/model_training.py:**  
  Defines the base classifiers and provides two separate functions for ensemble training:  
  - `train_bagging_ensemble`: Trains a bagging ensemble for each base classifier.
  - `train_stacking_ensemble`: Creates a stacking ensemble by combining all optimized base models.  
  The `train_all_ensembles` function trains bagging ensembles for all classifiers and a single stacking ensemble.

- **src/training/classifier_config.py:**  
  Manages classifier configurations and provides a registry system for easy classifier management.

### Evaluation
- **src/evaluation/evaluation.py:**  
  Contains functions for calculating evaluation metrics (accuracy, precision, recall, F1 score) for individual models and sets of models.

- **src/evaluation/evaluation_visualization.py:**  
  Combines plotting and reporting functionality. It includes functions for plotting error curves and converting evaluation results into pandas DataFrames.

- **src/evaluation/compute_errors.py:**  
  Provides a function to compute train and test errors for both optimized models and bagging ensembles.

### Utilities
- **src/utils/config.py:**  
  Centralized configuration management with dataclass-based settings.

- **src/utils/helpers.py:**  
  Contains common imports, utility functions (e.g., `get_project_root()`), and plotting utilities.

- **src/utils/logging.py:**  
  Centralized logging configuration with file and console handlers.

- **src/utils/logging_config.py:**  
  Advanced logging configuration options and utilities.

- **src/utils/interfaces.py:**  
  Defines abstract base classes and interfaces for the system.

- **src/utils/pipeline_orchestrator.py:**  
  Orchestrates the entire pipeline execution with proper error handling.

### Main Application
- **main.py:**  
  Orchestrates the entire pipeline:
  - Runs the data cleaning and preprocessing pipelines.
  - Loads and vectorizes the processed data.
  - Performs cross-validation.
  - Optimizes hyperparameters.
  - Trains bagging ensembles for each classifier and a stacking ensemble that combines all optimized models.
  - Evaluates each model and ensemble.
  - Computes and plots error curves.
  - Logs every step and saves all visualizations to `output/plots`.

## Logging and Visualizations

- **Logging:**  
  Every major step and output is logged to `output/log.txt` via the centralized logging module. This ensures that the pipeline execution is traceable and debuggable.

- **Visualizations:**  
  All generated visual outputs (e.g., cross-validation results, error curves, confusion matrices) are automatically saved in the `output/plots` directory.

## How to Run

### Prerequisites
1. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Development Dependencies (Optional):**  
   ```bash
   pip install -r dev-requirements.txt
   ```

### Running the Application
1. **Execute the Pipeline:**  
   ```bash
   paver run
   ```
   or
   ```bash
   python main.py --vectorizer tfidf --test-size 0.2
   ```

2. **Run Tests:**  
   ```bash
   paver test
   ```

### Command Line Options
- `--vectorizer`: Choose between 'count' or 'tfidf' (default: 'tfidf')
- `--test-size`: Test set size as fraction (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

## Recent Improvements

### Code Quality Enhancements
- **Test Suite Refactoring**: All 361 tests now follow SOLID principles with focused, single-purpose test functions
- **Improved Maintainability**: Each test function has only one or two assertions for better debugging
- **Clean Code**: Removed unnecessary comments and docstrings following minimal code style preferences
- **Bug Fixes**: Resolved critical logging configuration issues and import errors

### Performance Results
The system achieves excellent performance with ensemble methods:
- **Stacking Ensemble**: 88.13% Accuracy (best overall)
- **Support Vector Machine**: 87.19% Accuracy  
- **Logistic Regression**: 87.19% Accuracy
- **Multilayer Perceptron**: 86.56% Accuracy

### Code Coverage
- **94% Code Coverage** achieved across all modules
- **361 Tests Passing** with comprehensive test coverage
- **Zero Critical Issues** in the codebase

## Project Status

✅ **Fully Functional**: All components working correctly  
✅ **Well Tested**: Comprehensive test suite with 94% coverage  
✅ **Production Ready**: Robust error handling and logging  
✅ **Maintainable**: Clean, modular code following SOLID principles  

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass (`paver test`)
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.