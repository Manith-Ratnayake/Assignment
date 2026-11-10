import time
from dataclasses import replace
import joblib

from training.data import load_processed_data
from training.metrics import evaluate_splits
from training.ml_models import build_ml_model
from training.results import append_result, build_result_row, make_run_dir, print_summary, save_run

from dataclasses import replace
from training.grid import grid_combinations


def run_ml(config):
    """Train one scikit-learn style model (MLConfig), evaluate it and log the result."""
    splits, target_scaler, feature_config = load_processed_data(config.data_dir, config.evaluate_test)
    X_train, y_train = splits["train"]

    # 1. Build and train (config.seed is passed to models that use randomness)
    model = build_ml_model(config.model_name, config.model_params, config.seed)
    print(f"Experiment: {config.experiment_name} | Model: {config.model_name} {config.model_params}")
    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    # 2. Evaluate in Wh
    metrics, predictions = evaluate_splits(model.predict, splits, target_scaler)

    # 3. Save the run and log it
    run_dir = make_run_dir(config.output_dir, config.experiment_name)
    joblib.dump(model, run_dir / "model.joblib")
    save_run(run_dir, config, feature_config, metrics, predictions)
    append_result(config.output_dir, build_result_row(run_dir, config, feature_config, metrics, train_seconds))

    print_summary(metrics, run_dir)
    return metrics


def run_ml_grid(grid_path, base_config):
    """Run run_ml once for every combination in the grid file (e.g. ml.json).

    Every setting in the file goes into model_params. Everything else comes from base_config.
    """
    for model_name, params, run_name in grid_combinations(grid_path):
        config = replace(base_config, experiment_name=run_name, model_name=model_name,
                         model_params=params, notes=base_config.notes or "grid search")
        run_ml(config)