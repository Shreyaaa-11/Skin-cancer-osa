from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TrainConfig:
    image_size: int = 224
    num_classes: int = 7
    batch_size: int = 32
    stage1_epochs: int = 8
    stage2_epochs: int = 12
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-5
    backbone: str = "efficientnetb0"
    unfreeze_last_n_layers: int = 40
    class_names: List[str] = None
    model_output_path: Path = Path("backend/artifacts/model.keras")
    thresholds_output_path: Path = Path("backend/artifacts/thresholds.json")

    def __post_init__(self) -> None:
        if self.class_names is None:
            self.class_names = [
                "akiec",
                "bcc",
                "bkl",
                "df",
                "mel",
                "nv",
                "vasc",
            ]
