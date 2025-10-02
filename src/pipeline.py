#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt
from typing import Dict, Any
import json, time, hashlib
import pandas as pd
import numpy as np
from src.utils.json_utils import to_serializable

from sklearn.metrics import confusion_matrix, classification_report

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
from src.utils.logging_config import get_logger, setup_logging


class ModelPipeline:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.root = self._get_project_root()

    def _get_project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(self, preprocessed_data=None, vectorizer_type=None, test_size=None) -> Dict[str, Any]:
        vectorizer_type = vectorizer_type or config.DEFAULT_VECTORIZER
        test_size = test_size or config.TEST_SIZE

        setup_logging(config.get_log_file_path(self._get_project_root()))
        self.logger.info(f"Starting model pipeline with vectorizer={vectorizer_type}, test_size={test_size}")

        vectorizer, X_train_vec, X_test_vec, y_train, y_test, raw_data = load_and_vectorize_data(
            test_size=test_size,
            vectorizer_type=vectorizer_type,
            random_state=config.RANDOM_STATE
        )

        run_ts = time.strftime("%Y%m%d-%H%M%S")
        results_root = config.get_results_dir(self.root)
        results_dir = os.path.join(results_root, run_ts)
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        data_path, data_sha256 = None, None
        try:
            data_path = config.get_preprocessed_data_path(self.root)
            if os.path.exists(data_path):
                with open(data_path, "rb") as fh:
                    data_sha256 = hashlib.sha256(fh.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not hash preprocessed data: {e}")

        cv_results = cross_validate_models(
            X_train_vec, y_train, BASE_CLASSIFIERS, n_splits=config.CV_SPLITS
        )
        fig_cv, _ = plot_cv_results(cv_results)
        cv_plot_path = os.path.join(plots_dir, "cross_validation_metrics.png")
        fig_cv.savefig(cv_plot_path, bbox_inches="tight")
        plt.close(fig_cv)

        try:
            pd.DataFrame(cv_results).T.to_csv(os.path.join(results_dir, "cv_results.csv"))
        except Exception:
            with open(os.path.join(results_dir, "cv_results.json"), "w") as f:
                json.dump(cv_results, f, indent=2, default=to_serializable)
        else:
            with open(os.path.join(results_dir, "cv_results.json"), "w") as f:
                json.dump(cv_results, f, indent=2, default=to_serializable)

        best_models, best_params_dict = optimize_all_classifiers(
            BASE_CLASSIFIERS, HYPERPARAMS, X_train_vec, y_train
        )

        with open(os.path.join(results_dir, "best_params.json"), "w") as f:
            json.dump(best_params_dict, f, indent=2, default=to_serializable)

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

        with open(os.path.join(results_dir, "opt_results.json"), "w") as f:
            json.dump(opt_results, f, indent=2, default=to_serializable)
        with open(os.path.join(results_dir, "bag_results.json"), "w") as f:
            json.dump(bag_results, f, indent=2, default=to_serializable)
        with open(os.path.join(results_dir, "stack_result.json"), "w") as f:
            json.dump({"Stacking": stack_result}, f, indent=2, default=to_serializable)

        df_opt = create_results_dataframe(opt_results)
        df_bag = create_results_dataframe(bag_results)
        df_opt.to_csv(os.path.join(results_dir, "opt_results.csv"), index=False)
        df_bag.to_csv(os.path.join(results_dir, "bag_results.csv"), index=False)

        self.logger.info("\nOptimized-models DataFrame:")
        self.logger.info(f"\n{df_opt}")
        self.logger.info("\nBagging-ensembles DataFrame:")
        self.logger.info(f"\n{df_bag}")
        self.logger.info(f"\nStacking ensemble result: {stack_result}")

        y_pred_stack = stacking_ensemble.predict(X_test_vec)
        pd.DataFrame(
            {"y_true": y_test.reset_index(drop=True), "y_pred": pd.Series(y_pred_stack)}
        ).to_csv(os.path.join(results_dir, "stacking_predictions.csv"), index=False)

        cm = confusion_matrix(y_test, y_pred_stack)
        pd.DataFrame(cm).to_csv(os.path.join(results_dir, "stacking_confusion_matrix.csv"), index=False)

        report = classification_report(y_test, y_pred_stack, output_dict=True, zero_division=0)
        with open(os.path.join(results_dir, "stacking_classification_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=to_serializable)

        cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
        plot_confusion_matrix(
            y_test,
            y_pred_stack,
            labels=None,
            normalize=False,
            ax=cm_ax,
            cmap="Blues"
        )
        cm_plot_path = os.path.join(plots_dir, "stacking_confusion_matrix.png")
        cm_fig.savefig(cm_plot_path, bbox_inches="tight")
        plt.close(cm_fig)

        train_err, test_err = compute_errors(
            best_models,
            bagging_ensembles,
            stacking_ensemble,
            X_train_vec,
            X_test_vec,
            y_train,
            y_test
        )

        fig_err, _ = plot_error_curves(train_err, test_err)
        err_plot_path = os.path.join(plots_dir, "train_test_error_curves.png")
        fig_err.savefig(err_plot_path, bbox_inches="tight")
        plt.close(fig_err)

        with open(os.path.join(results_dir, "train_errors.json"), "w") as f:
            json.dump(train_err, f, indent=2, default=to_serializable)
        with open(os.path.join(results_dir, "test_errors.json"), "w") as f:
            json.dump(test_err, f, indent=2, default=to_serializable)

        manifest = {
            "timestamp": run_ts,
            "vectorizer": vectorizer_type,
            "test_size": test_size,
            "random_state": config.RANDOM_STATE,
            "data_path": data_path,
            "data_sha256": data_sha256,
            "plots": {
                "cv_metrics": cv_plot_path,
                "error_curves": err_plot_path,
                "stacking_confusion_matrix": cm_plot_path
            },
            "artifacts": [
                "cv_results.csv",
                "cv_results.json",
                "best_params.json",
                "opt_results.csv",
                "opt_results.json",
                "bag_results.csv",
                "bag_results.json",
                "stack_result.json",
                "stacking_predictions.csv",
                "stacking_confusion_matrix.csv",
                "stacking_classification_report.json",
                "train_errors.json",
                "test_errors.json"
            ]
        }
        with open(os.path.join(results_dir, "run_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=to_serializable)

        self.logger.info(f"\n✅ All artifacts saved under: {results_dir}")
        self.logger.info(f"   Plots saved under: {plots_dir}")

        return {
            "cv_results": cv_results,
            "best_models": best_models,
            "best_params": best_params_dict,
            "bagging_ensembles": bagging_ensembles,
            "stacking_ensemble": stacking_ensemble,
            "opt_results": opt_results,
            "bag_results": bag_results,
            "stack_result": stack_result,
            "train_errors": train_err,
            "test_errors": test_err,
            "results_dir": results_dir,
        }

    def _print_evaluation_results(self, opt_results, bag_results, stack_result):
        self.logger.info("\nOptimized base classifiers on test set:")
        for name, m in opt_results.items():
            self.logger.info(f"• {name:<25}  {m}")
        self.logger.info("\nBagging ensembles on test set:")
        for name, m in bag_results.items():
            self.logger.info(f"• {name:<25}  {m}")
        self.logger.info("\nStacking ensemble on test set:")
        self.logger.info(f"• Stacking                 {stack_result}")


model_pipeline = ModelPipeline()

def model_pipeline_func(vectorizer_type: str = "count", test_size: float = 0.2):
    return model_pipeline.run(
        vectorizer_type=vectorizer_type,
        test_size=test_size
    )
