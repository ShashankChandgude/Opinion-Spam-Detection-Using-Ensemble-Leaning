#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import matplotlib.pyplot as plt

from src.features.vectorization import load_and_vectorize_data
from src.training.cross_validation import cross_validate_models, plot_cv_results
from src.training.hyperparameter_optimization import optimize_all_classifiers, HYPERPARAMS
from src.training.model_training import train_all_ensembles, BASE_CLASSIFIERS
from src.evaluation.evaluation import evaluate_model, evaluate_models
from src.evaluation.evaluation_visualization import (
    plot_error_curves,
    create_results_dataframe,
    plot_confusion_matrix  # we will call this differently
)
from src.evaluation.compute_errors import compute_errors, compute_confusion_matrix


def model_pipeline(vectorizer_type: str = "count", test_size: float = 0.2):
    vectorizer, X_train_vec, X_test_vec, y_train, y_test, raw_data = \
        load_and_vectorize_data(test_size=test_size, vectorizer_type=vectorizer_type)

    cv_results = cross_validate_models(X_train_vec, y_train, BASE_CLASSIFIERS, n_splits=5)
    fig_cv, ax_cv = plot_cv_results(cv_results)

    plot_dir = os.path.join(os.getcwd(), "output", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    cv_path = os.path.join(plot_dir, "cross_validation_metrics.png")
    fig_cv.savefig(cv_path)
    plt.close(fig_cv)

    best_models, best_params_dict = optimize_all_classifiers(
        BASE_CLASSIFIERS, HYPERPARAMS, X_train_vec, y_train
    )

    print("\nOptimized model results on test set:")
    for name, model in best_models.items():
        metrics = evaluate_model(model, X_test_vec, y_test)
        print(f"• {name:<25}  {metrics}")

    bagging_ensembles, stacking_ensemble = train_all_ensembles(
        BASE_CLASSIFIERS,
        best_params_dict,
        X_train_vec,
        y_train,
        best_models
    )

    opt_results  = evaluate_models(best_models, X_test_vec, y_test)
    bag_results  = evaluate_models(bagging_ensembles, X_test_vec, y_test)
    stack_result = evaluate_model(stacking_ensemble,  X_test_vec, y_test)

    print("\nOptimized base classifiers on test set:")
    for name, m in opt_results.items():
        print(f"• {name:<25}  {m}")

    print("\nBagging ensembles on test set:")
    for name, m in bag_results.items():
        print(f"• {name:<25}  {m}")

    print("\nStacking ensemble on test set:")
    print(f"• Stacking        {stack_result}")

    train_err, test_err = compute_errors(
        best_models,
        bagging_ensembles,
        stacking_ensemble,
        X_train_vec,
        X_test_vec,
        y_train,
        y_test
    )

    fig_err, ax_err = plot_error_curves(train_err, test_err)
    err_path = os.path.join(plot_dir, "train_test_error_curves.png")
    fig_err.savefig(err_path)
    plt.close(fig_err)

    df_opt = create_results_dataframe(opt_results)
    df_bag = create_results_dataframe(bag_results)

    print("\nOptimized‐models DataFrame:")
    print(df_opt)

    print("\nBagging‐ensembles DataFrame:")
    print(df_bag)

    print("\nStacking ensemble result:")
    print(stack_result)

    y_pred_stack = stacking_ensemble.predict(X_test_vec)

    cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(
        y_test,           
        y_pred_stack,      
        labels=None,      
        normalize=False,   
        ax=cm_ax,
        cmap="Blues"
    )
    cm_path = os.path.join(plot_dir, "stacking_confusion_matrix.png")
    cm_fig.savefig(cm_path)
    plt.close(cm_fig)

    print(f"\n✅ All plots saved under: {plot_dir}\n")
