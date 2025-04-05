#!/usr/bin/env python
# coding: utf-8

from src.data_cleaning import pipeline as cleaning_pipeline
from src.preprocessing import pipeline as preprocessing_pipeline
from src.vectorization import load_and_vectorize_data
from src.cross_validation import run_cv_and_plot
from src.hyperparameter_optimization import optimize_all_classifiers, HYPERPARAMS
from src.model_training import train_all_ensembles, BASE_CLASSIFIERS
from src.evaluation import evaluate_model, evaluate_models
from src.evaluation_visualization import plot_error_curves, create_results_dataframe
from src.compute_errors import compute_errors

def main():
    
    cleaning_pipeline()
    preprocessing_pipeline()
    
    vectorizer, X_train_vec, X_test_vec, y_train, y_test, data = load_and_vectorize_data()
    
    base_classifiers = BASE_CLASSIFIERS
    
    cv_results = run_cv_and_plot(X_train_vec, y_train, base_classifiers, n_splits=5)
   
    best_models, best_params_dict = optimize_all_classifiers(BASE_CLASSIFIERS, HYPERPARAMS, X_train_vec, y_train)
    for name, model in best_models.items():
        print(f"Optimized results for {name}: {evaluate_model(model, X_test_vec, y_test)}")
    
    bagging_ensembles, stacking_ensemble = train_all_ensembles(BASE_CLASSIFIERS, best_params_dict, X_train_vec, y_train, best_models)
    
    results_opt = evaluate_models(best_models, X_test_vec, y_test)
    bagging_results = evaluate_models(bagging_ensembles, X_test_vec, y_test)
    stacking_result = evaluate_model(stacking_ensemble, X_test_vec, y_test)
    
    print("Optimized Models Results:")
    for name, metrics in results_opt.items():
        print(f"{name}: {metrics}")
    print("Bagging Ensemble Results:")
    for name, metrics in bagging_results.items():
        print(f"{name}: {metrics}")
    print("Stacking Ensemble Result:")
    print(stacking_result)
    
    train_errors, test_errors = compute_errors(best_models, bagging_ensembles, X_train_vec, X_test_vec, y_train, y_test)
    plot_error_curves(train_errors, test_errors)
   
    results_df_opt = create_results_dataframe(evaluate_models(best_models, X_test_vec, y_test))
    results_df_bag = create_results_dataframe(bagging_results)
    
    print("Optimized Models DataFrame:")
    print(results_df_opt)
    print("Bagging Ensembles DataFrame:")
    print(results_df_bag)
    print("Stacking Ensemble Result:")
    print(stacking_result)

if __name__ == "__main__":
    main()