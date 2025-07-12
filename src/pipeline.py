#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.features.vectorization import load_and_vectorize_data
from src.training.cross_validation import cross_validate_models, plot_cv_results
from src.training.hyperparameter_optimization import optimize_all_classifiers, HYPERPARAMS
from src.training.model_training import train_all_ensembles, BASE_CLASSIFIERS
from src.evaluation.evaluation import evaluate_model, evaluate_models
from src.evaluation.evaluation_visualization import (
    plot_error_curves,
    create_results_dataframe,
    plot_confusion_matrix
)
from src.evaluation.compute_errors import compute_errors
from src.utils.config import config
from src.utils.logging_config import get_logger

class ModelPipeline:
    """Model pipeline that handles training, evaluation, and visualization."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.root = self._get_project_root()
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def run(self, preprocessed_data=None, vectorizer_type=None, test_size=None) -> Dict[str, Any]:
        """
        Run the complete model pipeline.
        
        Args:
            preprocessed_data: Preprocessed data DataFrame
            vectorizer_type: Type of vectorizer to use
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary containing model results
        """
        vectorizer_type = vectorizer_type or config.DEFAULT_VECTORIZER
        test_size = test_size or config.TEST_SIZE
        
        self.logger.info(f"Starting model pipeline with vectorizer={vectorizer_type}, test_size={test_size}")
        
        vectorizer, X_train_vec, X_test_vec, y_train, y_test, raw_data = \
            load_and_vectorize_data(
                test_size=test_size,
                vectorizer_type=vectorizer_type,
                random_state=config.RANDOM_STATE
            )
        
        cv_results = cross_validate_models(
            X_train_vec, y_train, BASE_CLASSIFIERS, n_splits=config.CV_SPLITS
        )
        fig_cv, ax_cv = plot_cv_results(cv_results)
        
        plot_dir = config.get_plots_dir(self.root)
        os.makedirs(plot_dir, exist_ok=True)
        cv_path = os.path.join(plot_dir, "cross_validation_metrics.png")
        fig_cv.savefig(cv_path)
        plt.close(fig_cv)
        
        best_models, best_params_dict = optimize_all_classifiers(
            BASE_CLASSIFIERS, HYPERPARAMS, X_train_vec, y_train
        )
        
        self.logger.info("\nOptimized model results on test set:")
        for name, model in best_models.items():
            metrics = evaluate_model(model, X_test_vec, y_test)
            self.logger.info(f"• {name:<25}  {metrics}")
        
        bagging_ensembles, stacking_ensemble = train_all_ensembles(
            BASE_CLASSIFIERS,
            best_params_dict,
            X_train_vec,
            y_train,
            best_models
        )
        
        opt_results = evaluate_models(best_models, X_test_vec, y_test)
        bag_results = evaluate_models(bagging_ensembles, X_test_vec, y_test)
        stack_result = evaluate_model(stacking_ensemble, X_test_vec, y_test)
        
        self._print_evaluation_results(opt_results, bag_results, stack_result)
        
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
        
        self.logger.info("\nOptimized-models DataFrame:")
        self.logger.info(f"\n{df_opt}")
        
        self.logger.info("\nBagging-ensembles DataFrame:")
        self.logger.info(f"\n{df_bag}")
        
        self.logger.info(f"\nStacking ensemble result: {stack_result}")
        
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
        
        self.logger.info(f"\n✅ All plots saved under: {plot_dir}")
        
        return {
            'cv_results': cv_results,
            'best_models': best_models,
            'best_params': best_params_dict,
            'bagging_ensembles': bagging_ensembles,
            'stacking_ensemble': stacking_ensemble,
            'opt_results': opt_results,
            'bag_results': bag_results,
            'stack_result': stack_result,
            'train_errors': train_err,
            'test_errors': test_err
        }
    
    def _print_evaluation_results(self, opt_results, bag_results, stack_result):
        """Print evaluation results in a formatted way."""
        self.logger.info("\nOptimized base classifiers on test set:")
        for name, m in opt_results.items():
            self.logger.info(f"• {name:<25}  {m}")
        
        self.logger.info("\nBagging ensembles on test set:")
        for name, m in bag_results.items():
            self.logger.info(f"• {name:<25}  {m}")
        
        self.logger.info("\nStacking ensemble on test set:")
        self.logger.info(f"• Stacking        {stack_result}")

model_pipeline = ModelPipeline()
def model_pipeline_func(vectorizer_type: str = "count", test_size: float = 0.2):
    """Legacy model_pipeline function for backward compatibility."""
    return model_pipeline.run(
        vectorizer_type=vectorizer_type,
        test_size=test_size
    )
