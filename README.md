# Opinion Spam Detection Using Ensemble Learning

## Introduction

This project implements a machine learning pipeline for detecting deceptive reviews using ensemble learning methods. The system combines multiple base classifiers through bagging and stacking techniques to improve classification performance.

The pipeline processes text data through cleaning, preprocessing, feature engineering, and model training stages, with comprehensive logging and experiment tracking throughout.

## Dataset

**Deceptive Opinion Corpus**
- **Source**: [Deceptive Opinion Spam Dataset](https://www.cs.cornell.edu/~llee/deceptive-opinion-spam/)
- **Size**: 1,600 reviews (800 deceptive, 800 truthful)
- **Format**: CSV with hotel review text and labels
- **Usage**: Place the dataset file as `data/raw/deceptive-opinion-corpus.csv`

To obtain the dataset:
1. Visit the Cornell University research page linked above
2. Download the deceptive opinion spam corpus
3. Extract and place the CSV file in the `data/raw/` directory
4. Ensure the file is named `deceptive-opinion-corpus.csv`

## Project Structure

```
Opinion-Spam-Detection-Using-Ensemble-Learning/
├── main.py                          # Main application entry point
├── pavement.py                      # Build automation and task runner
├── pytest.ini                       # Test configuration
├── requirements.txt                 # Production dependencies
├── dev-requirements.txt             # Development dependencies
├── README.md                        # This documentation
├── data/
│   ├── raw/
│   │   └── deceptive-opinion-corpus.csv  # Original dataset (user-provided)
│   └── processed/
│       ├── cleaned_data.csv         # Cleaned dataset
│       └── preprocessed_data.csv     # Preprocessed dataset
├── src/                             # Source code modules
│   ├── data/                        # Data processing modules
│   ├── evaluation/                  # Model evaluation modules
│   ├── features/                    # Feature engineering modules
│   ├── training/                    # Model training modules
│   ├── utils/                       # Utility modules
│   └── pipeline.py                  # Main ML pipeline
├── tests/                           # Test suite
└── output/
    ├── log.txt                      # Main execution log
    ├── data_preprocessing/          # Preprocessing visualizations
    └── runs/                        # Timestamped experiment results
        └── 20251002-025246/         # Latest complete run
            ├── plots/               # Visualizations
            ├── *.csv                # Performance metrics
            ├── *.json               # Structured results
            └── run_manifest.json    # Run metadata
```

## Architecture

### Data Processing Pipeline
- **Data Cleaning**: Removes duplicates, handles missing values, basic validation
- **Text Preprocessing**: Tokenization, stopword removal, stemming, special character handling
- **Feature Engineering**: TF-IDF vectorization with configurable parameters

### Model Training
- **Base Classifiers**: Logistic Regression, Decision Trees, Random Forest, K-NN, Naive Bayes, SVM, MLP
- **Ensemble Methods**: 
  - Bagging: Applied to each base classifier
  - Stacking: Meta-learner combining all optimized models
- **Hyperparameter Optimization**: Randomized search with cross-validation

### Evaluation
- **Cross-validation**: 5-fold stratified validation
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrices, error curves, performance plots

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Full pipeline execution
paver run

# Direct execution with options
python main.py --vectorizer tfidf --test-size 0.2 --random-state 42
```

### Command Line Options
- `--vectorizer`: Choose 'count' or 'tfidf' (default: 'tfidf')
- `--test-size`: Test set size as fraction (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

### Running Tests
```bash
paver test
```

## Results

### Latest Run Performance (2025-10-02-025246)

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| **Stacking Ensemble** | **88.13%** | **88.22%** | **88.13%** | **88.12%** |
| Support Vector Machine | 87.19% | 87.20% | 87.19% | 87.19% |
| Logistic Regression | 87.19% | 87.22% | 87.19% | 87.18% |
| Multilayer Perceptron | 85.94% | 86.05% | 85.94% | 85.93% |
| Random Forest | 84.38% | 84.38% | 84.38% | 84.37% |
| Multinomial Naive Bayes | 85.31% | 86.06% | 85.31% | 85.24% |
| K-Nearest Neighbors | 78.75% | 79.30% | 78.75% | 78.65% |
| Decision Tree | 67.81% | 68.12% | 67.81% | 67.68% |

### Experiment Tracking
- **Timestamped Runs**: Each execution creates a unique directory under `output/runs/`
- **Comprehensive Logging**: All operations logged to `output/log.txt`
- **Artifact Management**: Models, metrics, and visualizations saved systematically
- **Reproducibility**: SHA256 checksums and run manifests for experiment tracking

## Technical Details

### Test Coverage
- **Overall Coverage**: 97% (673 statements, 20 missing)
- **Test Count**: 383 tests
- **Execution Time**: ~32 seconds for full test suite
- **Integration Tests**: Comprehensive mocking for fast CI/CD pipelines

### Dependencies
- **Core ML**: scikit-learn, numpy, pandas
- **Text Processing**: NLTK, wordcloud
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, pytest-cov
- **Build**: paver

## Project Status

- **Functional**: All components working correctly  
- **Tested**: Comprehensive test suite with 97% coverage  
- **Documented**: Clear structure and usage instructions  
- **Reproducible**: Consistent results with proper experiment tracking  

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass (`paver test`)
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure you have proper rights to use the dataset according to its original terms.