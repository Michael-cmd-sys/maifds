import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor, nn, ops, dataset as ds

from config import (
    TRAINING_TABLE_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from model import ProactiveWarningModel


def load_training_data():
    df = pd.read_parquet(TRAINING_TABLE_PATH)

    # ensure all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training table: {missing}")

    X = df[FEATURE_COLUMNS].astype("float32").values
    y = df[LABEL_COLUMN].astype("float32").values.reshape(-1, 1)

    return X, y


def train_val_split(X, y, val_ratio=0.15, seed=RANDOM_SEED):
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    split = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def make_dataset(X, y, batch_size, shuffle=True):
    data = {"features": X, "labels": y}
    dataset = ds.NumpySlicesDataset(data=data, column_names=["features", "labels"], shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def standardize_features(X_train, X_val):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    return X_train_norm, X_val_norm, mean, std


def main():
    # CPU context (your env is CPU-only MindSpore)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    X, y = load_training_data()
    X_train, y_train, X_val, y_val = train_val_split(X, y)

    X_train, X_val, mean, std = standardize_features(X_train, X_val)

    print(f"Loaded training data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    input_dim = X_train.shape[1]
    model = ProactiveWarningModel(input_dim=input_dim)

    loss_fn = nn.BCELoss(reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)

    # Datasets
    train_dataset = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_dataset = make_dataset(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    # Forward + grad
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    # Validation
    def evaluate():
        model.set_train(False)
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in val_dataset.create_dict_iterator():
            features = Tensor(batch["features"], ms.float32)
            labels = Tensor(batch["labels"], ms.float32)
            preds = model(features)
            loss = loss_fn(preds, labels)

            total_loss += float(loss.asnumpy()) * labels.shape[0]
            total += labels.shape[0]

            preds_bin = (preds.asnumpy() >= 0.5).astype(np.float32)
            correct += (preds_bin == labels.asnumpy()).sum()

        avg_loss = total_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.set_train(True)
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataset.create_dict_iterator():
            features = Tensor(batch["features"], ms.float32)
            labels = Tensor(batch["labels"], ms.float32)

            loss = train_step(features, labels)
            epoch_loss += float(loss.asnumpy())
            num_batches += 1

        train_loss = epoch_loss / max(1, num_batches)
        val_loss, val_acc = evaluate()

        print(
            f"Epoch {epoch}/{EPOCHS} - "
            f"train_loss={train_loss:.4f} - val_loss={val_loss:.4f} - val_acc={val_acc:.4f}"
        )

    # Save model + normalization stats
    ckpt_name = "proactive_warning_mlp.ckpt"
    ms.save_checkpoint(model, ckpt_name)
    print(f"Model checkpoint saved as {ckpt_name}")

    # Save normalization statistics for inference
    np.save("feature_mean.npy", mean)
    np.save("feature_std.npy", std)
    print("Saved feature normalization stats (feature_mean.npy, feature_std.npy)")


if __name__ == "__main__":
    main()
