import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import nn, Tensor, ops

from config import (
    TRAINING_TABLE_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from model import CallTriggeredDefenseModel


def load_training_data():
    """
    Load the processed parquet table and return X, y as numpy arrays.
    """
    df = pd.read_parquet(TRAINING_TABLE_PATH)

    # Drop rows with missing values in our selected columns
    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])

    X = df[FEATURE_COLUMNS].astype("float32").values
    y = df[LABEL_COLUMN].astype("float32").values.reshape(-1, 1)

    return X, y


def create_dataset(X, y, batch_size, shuffle=True):
    """
    Wrap numpy arrays into a MindSpore dataset.
    """
    inputs = Tensor(X, ms.float32)
    labels = Tensor(y, ms.float32)

    dataset = ms.dataset.NumpySlicesDataset(
        {"features": inputs, "labels": labels},
        shuffle=shuffle,
    )
    dataset = dataset.batch(batch_size)
    return dataset


def train():
    ms.set_seed(RANDOM_SEED)

    # Load data
    X, y = load_training_data()
    print(f"Loaded training data: X shape = {X.shape}, y shape = {y.shape}")

    # Simple train/val split
    n = X.shape[0]
    val_size = int(0.15 * n)
    train_size = n - val_size

    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_ds = create_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds = create_dataset(X_val, y_val, BATCH_SIZE, shuffle=False)

    # Model
    input_dim = X.shape[1]
    model = CallTriggeredDefenseModel(input_dim=input_dim)

    # Loss & optimizer
    loss_fn = nn.BCELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.set_train()

        total_loss = 0.0
        num_batches = 0

        for batch in train_ds.create_dict_iterator():
            features = batch["features"]
            labels = batch["labels"]

            def forward_fn(x, y):
                preds = model(x)
                loss = loss_fn(preds, y)
                return loss, preds

            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, _), grads = grad_fn(features, labels)
            optimizer(grads)

            total_loss += float(loss.asnumpy())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Simple validation: compute average BCE on val set
        model.set_train(False)
        val_loss = 0.0
        val_batches = 0
        for batch in val_ds.create_dict_iterator():
            features = batch["features"]
            labels = batch["labels"]
            preds = model(features)
            loss = loss_fn(preds, labels)
            val_loss += float(loss.asnumpy())
            val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)

        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss={avg_loss:.4f} - val_loss={avg_val_loss:.4f}")

    # Save model parameters
    ms.save_checkpoint(model, "call_triggered_defense_mlp.ckpt")
    print("Model checkpoint saved as call_triggered_defense_mlp.ckpt")


if __name__ == "__main__":
    train()
