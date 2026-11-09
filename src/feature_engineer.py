from preprocessing.data_io import load_data, save_scalers, save_sequences
from preprocessing.cleaning import clip_outliers, compute_iqr_bounds
from preprocessing.features import check_no_missing, create_features, drop_warmup_rows
from preprocessing.splitting import get_split_indices, split_data
from preprocessing.scaling import apply_scalers, fit_scalers
from preprocessing.sequences import create_split_sequences


def main(config):
    """Run the full preprocessing pipeline described by config."""
    target = config["target"]
    train_ratio, val_ratio = config["train_ratio"], config["val_ratio"]

    # 1. Load, then clip outliers using bounds from the training portion only
    df = load_data(config["data_path"])
    train_end, _ = get_split_indices(len(df), train_ratio, val_ratio)
    lower, upper = compute_iqr_bounds(df[target].iloc[:train_end])
    df = clip_outliers(df, target, lower, upper)

    # 2. Build features and drop rows without enough history
    df = create_features(df, config)
    df = drop_warmup_rows(df, config["fixed_warmup"])
    check_no_missing(df)

    # 3. Chronological split
    train_end, val_end = get_split_indices(len(df), train_ratio, val_ratio)
    splits = split_data(df, train_end, val_end)
    for name, split in splits.items():
        print(f"{name.capitalize()}: {len(split)}")

    # 4. Scale (fitted on train only)
    feature_scaler, target_scaler = fit_scalers(
        splits["train"], target, config["feature_scaler"], config.get("target_scaler", "standard")
    )
    scaled = apply_scalers(splits, target, feature_scaler, target_scaler)

    # 5. Sliding windows for the sequence models
    sequences = create_split_sequences(scaled, config["sequence_length"])
    for name, (X, y) in sequences.items():
        print(f"{name.capitalize()} X: {X.shape} | y: {y.shape}")

    # 6. Save
    save_sequences(sequences, config["output_dir"])
    save_scalers(feature_scaler, target_scaler, config["output_dir"])
    
    print("Finished the data preprocessing step")
