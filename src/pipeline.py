from src.features.vectorization import load_and_vectorize_data
from src.training.cross_validation import run_cv_and_plot
from src.training.hyperparameter_optimization import optimize_all_classifiers, HYPERPARAMS
from src.training.model_training import train_all_ensembles, BASE_CLASSIFIERS
from src.evaluation.evaluation import evaluate_model, evaluate_models
from src.evaluation.evaluation_visualization import plot_error_curves, create_results_dataframe
from src.evaluation.compute_errors import compute_errors


def model_pipeline():
    """Full vectorization, training, evaluation and error plotting pipeline."""
    _, X_train_vec, X_test_vec, y_train, y_test, _ = load_and_vectorize_data()
    run_cv_and_plot(X_train_vec, y_train, BASE_CLASSIFIERS)

    best_models, best_params = optimize_all_classifiers(BASE_CLASSIFIERS, HYPERPARAMS, X_train_vec, y_train)
    for name, model in best_models.items():
        print(f"Optimized results for {name}: {evaluate_model(model, X_test_vec, y_test)}")
    
    bagging_ensembles, stacking_ensemble = train_all_ensembles(BASE_CLASSIFIERS, best_params, X_train_vec, y_train, best_models)
    
    opt_results = evaluate_models(best_models, X_test_vec, y_test)
    bag_results = evaluate_models(bagging_ensembles, X_test_vec, y_test)
    stack_result = evaluate_model(stacking_ensemble, X_test_vec, y_test)

    print("Optimized Models Results:")
    for name, m in opt_results.items():
        print(f"{name}: {m}")
    print("Bagging Ensemble Results:")
    for name, m in bag_results.items():
        print(f"{name}: {m}")
    print("Stacking Ensemble Result:")
    print(stack_result)

    train_err, test_err = compute_errors(best_models, bagging_ensembles, stacking_ensemble, X_train_vec, X_test_vec, y_train, y_test)
    plot_error_curves(train_err, test_err)

    df_opt = create_results_dataframe(opt_results)
    df_bag = create_results_dataframe(bag_results)
    print("Optimized Models DataFrame:")
    print(df_opt)
    print("Bagging Ensembles DataFrame:")
    print(df_bag)
    print("Stacking Ensemble Result:")
    print(stack_result)