import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Feature settings that are too long or not useful as results.csv columns
SKIPPED_FEATURE_KEYS = {"feature_columns", "data_path", "output_dir", "target"}


def make_run_dir(output_dir, experiment_name):
    """Create output_dir/<date_time>_<experiment_name>/ so runs never overwrite each other."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{stamp}_{experiment_name}"
    run_dir.mkdir(parents=True)
    return run_dir


def save_run(run_dir, config, feature_config, metrics, predictions, history=None):
    """Save everything about one run to its folder: settings, metrics, predictions and epoch history."""
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "feature_config.json", feature_config)
    write_json(run_dir / "metrics.json", metrics)

    for split, (actual, predicted) in predictions.items():
        if split != "train":
            pd.DataFrame({"actual": actual, "predicted": predicted}).to_csv(run_dir / f"predictions_{split}.csv", index=False)

    if history:
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)


def build_result_row(run_dir, config, feature_config, metrics, train_seconds, best_epoch=None):
    """Flatten one run into a single results.csv row: metrics first, then model and feature settings."""
    settings = asdict(config)
    row = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": settings.pop("experiment_name"),
        "model": settings.pop("model_name"),
    }

    for split in ("val", "test", "train"):
        for metric, value in metrics.get(split, {}).items():
            row[f"{split}_{metric}"] = round(value, 4)

    row["best_epoch"] = best_epoch
    row["train_seconds"] = round(train_seconds, 1)
    row["notes"] = settings.pop("notes")
    row["run_dir"] = run_dir.name

    for key in ("data_dir", "output_dir"):
        settings.pop(key)
    row.update({key: as_cell(value) for key, value in settings.items()})
    row.update({f"feat_{key}": as_cell(value) for key, value in feature_config.items()
                if key not in SKIPPED_FEATURE_KEYS})
    return row


def append_result(output_dir, row):
    """Add one row to output_dir/results.csv. ML and DL rows share the file; unused columns stay blank."""
    path = Path(output_dir) / "results.csv"
    table = pd.DataFrame([row])
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    table.to_csv(path, index=False)
    return path


def print_summary(metrics, run_dir):
    """Print one line of metrics per split and where the run was saved."""
    for split, values in metrics.items():
        print(f"{split.capitalize():5s} | MAE {values['mae']:.2f} Wh | RMSE {values['rmse']:.2f} Wh"
              f" | MAPE {values['mape']:.1f}% | R² {values['r2']:.3f}")
    print(f"Saved to: {run_dir}")


def as_cell(value):
    """Store lists and dicts as JSON text so they fit in one CSV cell."""
    return json.dumps(value) if isinstance(value, (list, tuple, dict)) else value


def write_json(path, data):
    """Write data to path as indented JSON."""
    Path(path).write_text(json.dumps(data, indent=2))
