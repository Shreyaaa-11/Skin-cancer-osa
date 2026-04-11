from typing import Literal

import tensorflow as tf
from tensorflow.keras import layers, models


def _build_backbone(
    backbone_name: Literal["efficientnetb0", "resnet50"],
    input_shape=(224, 224, 3),
) -> tf.keras.Model:
    if backbone_name == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    if backbone_name == "resnet50":
        return tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def build_skin_cancer_model(
    backbone_name: str = "efficientnetb0",
    input_shape=(224, 224, 3),
    num_classes: int = 7,
    dense_units: int = 256,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    backbone = _build_backbone(backbone_name, input_shape=input_shape)
    backbone.trainable = False

    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="skin_cancer_classifier")


def unfreeze_for_finetune(
    model: tf.keras.Model,
    unfreeze_last_n_layers: int = 40,
) -> None:
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("Backbone model not found for fine-tuning.")

    for layer in backbone.layers[:-unfreeze_last_n_layers]:
        layer.trainable = False
    for layer in backbone.layers[-unfreeze_last_n_layers:]:
        layer.trainable = True
