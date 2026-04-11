import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ml.config.settings import TrainConfig
from ml.models.efficientnet_model import build_skin_cancer_model, unfreeze_for_finetune


def make_dataset(data_dir: Path, image_size: int, batch_size: int, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def preprocess_dataset(ds):
    return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)


def compile_model(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def main(args):
    cfg = TrainConfig(backbone=args.backbone)
    data_root = Path(args.data_root)
    train_ds = preprocess_dataset(
        make_dataset(data_root / "train", cfg.image_size, cfg.batch_size, shuffle=True)
    )
    val_ds = preprocess_dataset(
        make_dataset(data_root / "val", cfg.image_size, cfg.batch_size, shuffle=False)
    )
    test_ds = preprocess_dataset(
        make_dataset(data_root / "test", cfg.image_size, cfg.batch_size, shuffle=False)
    )

    train_labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(cfg.num_classes),
        y=train_labels,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

    model = build_skin_cancer_model(
        backbone_name=cfg.backbone,
        input_shape=(cfg.image_size, cfg.image_size, 3),
        num_classes=cfg.num_classes,
    )

    compile_model(model, lr=cfg.stage1_lr)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage1_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    unfreeze_for_finetune(model, unfreeze_last_n_layers=cfg.unfreeze_last_n_layers)
    compile_model(model, lr=cfg.stage2_lr)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage2_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    cfg.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.model_output_path)

    Path("ml/data").mkdir(parents=True, exist_ok=True)
    y_true_test = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    probs_test = model.predict(test_ds)
    np.savez("ml/data/test_predictions.npz", y_true=y_true_test, probs=probs_test)

    report = {"model_path": str(cfg.model_output_path), "class_weights": class_weight_dict}
    with open("ml/data/train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnetb0", choices=["efficientnetb0", "resnet50"])
    main(parser.parse_args())
