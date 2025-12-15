import os
import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor, nn, dataset as ds

from .config import (
    TRAINING_TABLE_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from .model import ClickTxLinkModel


def _apply_transforms_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same transforms used by inference."""
    df = df.copy()

    if "url_hash_numeric" in df.columns:
        s = df["url_hash_numeric"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["url_hash_numeric"] = np.log1p(np.clip(s.to_numpy(dtype=np.float64), 0, None))

    #recommended: log time features to reduce extreme scale
    for col in ["time_between_click_and_tx", "time_since_last_tx_seconds"]:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[col] = np.log1p(np.clip(s.to_numpy(dtype=np.float64), 0, None))

    return df


def load_training_data():
    df = pd.read_parquet(TRAINING_TABLE_PATH).copy()

    # ensure all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in training table: {missing}")

    #sanitize
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    #apply transforms
    df = _apply_transforms_df(df)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

    return X, y


def train_val_split(X, y, val_ratio=0.15, seed=RANDOM_SEED):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def make_dataset(X, y, batch_size, shuffle=True):
    data = {"features": X, "labels": y}
    dataset = ds.NumpySlicesDataset(data=data, column_names=["features", "labels"], shuffle=shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def standardize_features(X_train, X_val):
    #compute stats in float64 to avoid overflow
    mean = X_train.astype(np.float64).mean(axis=0, keepdims=True)
    std = X_train.astype(np.float64).std(axis=0, keepdims=True)

    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    #final safety
    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return X_train_norm, X_val_norm, mean, std


def main():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    X, y = load_training_data()
    X_train, y_train, X_val, y_val = train_val_split(X, y)

    X_train, X_val, mean, std = standardize_features(X_train, X_val)

    print(f"Loaded training data: X shape={X.shape}, y shape={y.shape}")
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")

    model = ClickTxLinkModel(input_dim=X_train.shape[1])

    #stable loss for logits
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)

    train_dataset = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_dataset = make_dataset(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    def forward_fn(data, label):
        logits = model(data)          # logits
        loss = loss_fn(logits, label)
        return loss, logits

    weights = model.trainable_params()
    grad_fn = ms.value_and_grad(forward_fn, None, weights=weights, has_aux=True)


    #gradient clipping to stop NaN weights
    def _clip_grads(grads, clip_norm=1.0):
        total_norm = 0.0
        for g in grads:
            arr = g.asnumpy()
            total_norm += float((arr * arr).sum())
        total_norm = float(np.sqrt(total_norm))
        if total_norm > clip_norm and total_norm > 0:
            scale = clip_norm / total_norm
            grads = [g * scale for g in grads]
        return grads

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)

        # ✅ Important: MindSpore Adam in GRAPH_MODE prefers tuple grads
        if isinstance(grads, list):
            grads = tuple(grads)

        optimizer(grads)
        return loss


    def evaluate():
        model.set_train(False)
        total_loss = 0.0
        total = 0
        correct = 0

        for batch in val_dataset.create_dict_iterator():
            features = Tensor(batch["features"], ms.float32)
            labels = Tensor(batch["labels"], ms.float32)

            logits = model(features)
            loss = loss_fn(logits, labels)

            probs = 1.0 / (1.0 + ms.ops.Exp()(-logits))  # sigmoid(logits)
            preds_bin = (probs.asnumpy() >= 0.5).astype(np.float32)

            total_loss += float(loss.asnumpy()) * labels.shape[0]
            total += labels.shape[0]
            correct += (preds_bin == labels.asnumpy()).sum()

        avg_loss = total_loss / total if total else 0.0
        acc = correct / total if total else 0.0
        return avg_loss, acc

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

        print(f"Epoch {epoch}/{EPOCHS} - train_loss={train_loss:.4f} - val_loss={val_loss:.4f} - val_acc={val_acc:.4f}")

    #save checkpoint into the feature src directory (same place inference loads from)
    ckpt_path = os.path.join(os.path.dirname(__file__), "click_tx_link_model.ckpt")
    ms.save_checkpoint(model, ckpt_path)
    print(f"Model checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
