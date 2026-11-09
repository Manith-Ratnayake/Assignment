from preprocessing.data_io import load_data, save_feature_config, save_scalers, save_splits
from preprocessing.cleaning import clip_outliers, compute_iqr_bounds
from preprocessing.features import check_no_missing, create_features, drop_warmup_rows
from preprocessing.splitting import split_data
from preprocessing.scaling import apply_scalers, fit_scalers


def main(config):
    """Run the full preprocessing pipeline described by config."""
    target = config["target"]
    train_ratio, val_ratio = config["train_ratio"], config["val_ratio"]

    # 1. Load, then clip outliers using bounds from the training portion only
    df = load_data(config["data_path"])
    train_rows = split_data(df, train_ratio, val_ratio)["train"]
    lower, upper = compute_iqr_bounds(train_rows[target])
    df = clip_outliers(df, target, lower, upper)

    # 2. Build features and drop rows without enough history
    df = create_features(df, config)
    df = drop_warmup_rows(df, config["fixed_warmup"])
    check_no_missing(df)
    print(f"Features: {df.shape[1] - 1}")

    # 3. Chronological split
    splits = split_data(df, train_ratio, val_ratio)
    for name, split in splits.items():
        print(f"{name.capitalize()}: {len(split)}")

    # 4. Scale (fitted on train only)
    feature_scaler, target_scaler = fit_scalers(
        splits["train"], target, config["feature_scaler"], config.get("target_scaler", "standard")
    )
    scaled = apply_scalers(splits, target, feature_scaler, target_scaler)

    # 5. Save the scaled table, the scalers and a record of the feature settings.
    #    Windows for the sequence models are cut later, during deep learning training.
    save_splits(scaled, config["output_dir"])
    save_scalers(feature_scaler, target_scaler, config["output_dir"])
    save_feature_config(config, df.columns.drop(target), config["output_dir"])

    print("Finished the data preprocessing step")
