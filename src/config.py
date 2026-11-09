from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainConfig:
    model_name: str = "gru"
    model_params: dict[str, Any] = field(default_factory=dict)
    data_path: str = "data/energy_data_set.csv"
    output_dir: str = "outputs"
    target: str = "Appliances"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    sequence_length: int = 144
    include_interactions: bool = False
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 5
    seed: int = 42
    evaluate_test: bool = False