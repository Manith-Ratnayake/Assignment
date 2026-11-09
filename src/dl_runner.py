import time

import torch

from training.data import load_processed_data
from training.dl_models import build_dl_model
from training.metrics import compute_metrics, evaluate_splits, to_wh
from training.results import append_result, build_result_row, make_run_dir, print_summary, save_run
from training.torch_loop import create_loss, fit, get_device, make_train_loader, predict, set_seed
from training.windows import make_window_datasets


def run_dl(config):
    """Train one PyTorch sequence model (DLConfig) with early stopping, evaluate it and log the result."""
    set_seed(config.seed)
    device = get_device()
    splits, target_scaler, feature_config = load_processed_data(config.data_dir, config.evaluate_test)

    # 1. Cut the 2D table into windows of config.sequence_length rows
    windows = make_window_datasets(splits, config.sequence_length)
    train_set, _ = windows["train"]
    val_set, y_val = windows["val"]
    n_features = splits["train"][0].shape[1]

    # 2. Build the model, optimizer, loss and batches
    model = build_dl_model(config.model_name, n_features, config.model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = create_loss(config.loss, config.huber_delta)
    train_loader = make_train_loader(train_set, config.batch_size, config.seed)

    y_val_wh = to_wh(y_val, target_scaler)

    def score_val(val_pred):
        return compute_metrics(y_val_wh, to_wh(val_pred, target_scaler))

    print(f"Experiment: {config.experiment_name} | Model: {config.model_name} | Window: {config.sequence_length} | Device: {device}")
    print(model)

    # 3. Train with early stopping
    start = time.time()
    history, best_epoch = fit(
        model, train_loader, val_set, y_val, loss_fn, optimizer, device,
        config.epochs, config.patience, config.monitor, score_val, config.grad_clip,
    )
    train_seconds = time.time() - start

    # 4. Evaluate the best weights in Wh
    metrics, predictions = evaluate_splits(
        lambda dataset: predict(model, dataset, config.batch_size, device), windows, target_scaler
    )

    # 5. Save the run and log it
    run_dir = make_run_dir(config.output_dir, config.experiment_name)
    torch.save(model.state_dict(), run_dir / "model.pt")
    save_run(run_dir, config, feature_config, metrics, predictions, history)
    append_result(config.output_dir, build_result_row(run_dir, config, feature_config, metrics, train_seconds, best_epoch))

    print(f"Best epoch: {best_epoch}")
    print_summary(metrics, run_dir)
    return metrics
