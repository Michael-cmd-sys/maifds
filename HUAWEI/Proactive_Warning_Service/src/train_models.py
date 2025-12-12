"""
Train all 3 MindSpore models for Proactive Warning Service
Uses credit card fraud dataset (creditcard.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import logging

# MindSpore imports
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import TimeMonitor

# Import model architectures
from proactive_warning_service import (
    CohortSelectionModel,
    RiskUpliftModel,
    CampaignDetectionModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set MindSpore context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class CreditCardDataProcessor:
    """Process credit card fraud dataset for model training"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        logger.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} transactions")
        logger.info(f"Fraud cases: {self.df['Class'].sum()} ({self.df['Class'].mean()*100:.2f}%)")

    def prepare_cohort_selection_data(self) -> tuple:
        """
        Prepare data for Cohort Selection Model (vulnerable user identification)
        Features: User behavioral patterns (15 features)
        """
        logger.info("Preparing cohort selection data...")

        # Use fraud patterns to identify vulnerable user behaviors
        # Map transaction features to user behavior features
        features = []
        for idx, row in self.df.iterrows():
            # Simulate user features from transaction data
            user_features = [
                abs(row['V1']),  # recent_clicks proxy
                abs(row['V2']),  # recent_calls proxy
                1 if row['V3'] > 0 else 0,  # new_device indicator
                abs(row['V4'] * 100) % 365,  # device_age_days
                abs(int(row['V5'] * 10)) % 30,  # transaction_count_7d
                row['Amount'],  # avg_transaction_amount
                abs(row['Time'] / 86400) % 730,  # account_age_days
                abs(int(row['V6'] * 10)) % 20,  # risky_interactions_7d
                abs(row['V7']),  # additional behavioral feature
                abs(row['V8']),  # additional behavioral feature
                abs(row['V9']),  # additional behavioral feature
                abs(row['V10']),  # additional behavioral feature
                abs(row['V11']),  # additional behavioral feature
                abs(row['V12']),  # additional behavioral feature
                abs(row['V13']),  # additional behavioral feature
            ]
            features.append(user_features)

        X = np.array(features, dtype=np.float32)
        y = self.df['Class'].values.astype(np.float32)

        # Normalize features
        X = self.scaler.fit_transform(X)

        logger.info(f"Cohort data shape: X={X.shape}, y={y.shape}")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def prepare_risk_uplift_data(self) -> tuple:
        """
        Prepare data for Risk Uplift Model
        Features: User + campaign context (20 features)
        """
        logger.info("Preparing risk uplift data...")

        # Extended features including campaign context
        features = []
        for idx, row in self.df.iterrows():
            # Combine user features + campaign features
            uplift_features = [
                # User features (10)
                abs(row['V1']),
                abs(row['V2']),
                abs(row['V3']),
                abs(row['V4']),
                row['Amount'],
                abs(row['V5']),
                abs(row['V6']),
                abs(row['V7']),
                abs(row['V8']),
                abs(row['V9']),
                # Campaign context features (10)
                abs(row['V10']),  # campaign_volume proxy
                abs(row['V11']),  # campaign_anomaly_score proxy
                abs(row['V12']),  # time_since_campaign_start
                abs(row['V13']),  # user_exposure_count
                abs(row['V14']),  # similar_pattern_count
                abs(row['V15']),  # geographic_proximity
                abs(row['V16']),  # temporal_proximity
                abs(row['V17']),  # network_centrality
                abs(row['V18']),  # historical_susceptibility
                abs(row['V19']),  # current_risk_baseline
            ]
            features.append(uplift_features)

        X = np.array(features, dtype=np.float32)
        y = self.df['Class'].values.astype(np.float32)

        # Normalize features
        X = self.scaler.fit_transform(X)

        logger.info(f"Risk uplift data shape: X={X.shape}, y={y.shape}")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def prepare_campaign_detection_data(self, sequence_length: int = 24) -> tuple:
        """
        Prepare data for Campaign Detection Model (LSTM)
        Features: Time series of transaction patterns
        """
        logger.info(f"Preparing campaign detection sequences (length={sequence_length})...")

        # Create sequences from transaction data
        sequences = []
        labels = []

        # Use sliding window to create sequences
        feature_cols = [f'V{i}' for i in range(1, 11)]  # Use V1-V10 for temporal features

        for i in range(len(self.df) - sequence_length):
            # Extract sequence of transactions
            seq = self.df.iloc[i:i+sequence_length][feature_cols].values
            sequences.append(seq)
            # Label is 1 if any transaction in next window is fraud
            labels.append(float(self.df.iloc[i:i+sequence_length]['Class'].max()))

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        logger.info(f"Campaign sequences shape: X={X.shape}, y={y.shape}")
        logger.info(f"Campaign anomalies: {y.sum()} ({y.mean()*100:.2f}%)")

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_cohort_model(X_train, y_train, X_test, y_test, config):
    """Train Cohort Selection Model"""
    logger.info("=" * 60)
    logger.info("Training Cohort Selection Model")
    logger.info("=" * 60)

    # Create model
    model = CohortSelectionModel(input_size=15, hidden_sizes=[64, 32, 16])

    # Define loss and optimizer
    loss_fn = nn.BCELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config['learning_rate'])

    # Convert to MindSpore tensors
    train_data = Tensor(X_train, ms.float32)
    train_labels = Tensor(y_train.reshape(-1, 1), ms.float32)
    test_data = Tensor(X_test, ms.float32)
    test_labels = Tensor(y_test.reshape(-1, 1), ms.float32)

    # Training loop
    epochs = config['epochs']
    batch_size = config['batch_size']

    model.set_train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            # Forward pass
            def forward_fn(data, labels):
                logits = model(data)
                loss = loss_fn(logits, labels)
                return loss

            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
            loss, grads = grad_fn(batch_data, batch_labels)
            optimizer(grads)

            total_loss += loss.asnumpy()

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 5 == 0:
            # Evaluate on test set
            model.set_train(False)
            test_pred = model(test_data)
            test_loss = loss_fn(test_pred, test_labels)

            # Calculate accuracy
            predictions = (test_pred.asnumpy() > 0.5).astype(int)
            accuracy = (predictions == y_test.reshape(-1, 1)).mean()

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.4f}")
            model.set_train()

    # Save model
    model_path = config['models']['cohort_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ms.save_checkpoint(model, model_path)
    logger.info(f"✓ Cohort model saved to {model_path}")

    return model


def train_risk_uplift_model(X_train, y_train, X_test, y_test, config):
    """Train Risk Uplift Model"""
    logger.info("=" * 60)
    logger.info("Training Risk Uplift Model")
    logger.info("=" * 60)

    # Create model
    model = RiskUpliftModel(input_size=20, hidden_sizes=[128, 64, 32])

    # Define loss and optimizer
    loss_fn = nn.BCELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config['learning_rate'])

    # Convert to MindSpore tensors
    train_data = Tensor(X_train, ms.float32)
    train_labels = Tensor(y_train.reshape(-1, 1), ms.float32)
    test_data = Tensor(X_test, ms.float32)
    test_labels = Tensor(y_test.reshape(-1, 1), ms.float32)

    # Training loop
    epochs = config['epochs']
    batch_size = config['batch_size']

    model.set_train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            # Forward pass
            def forward_fn(data, labels):
                logits = model(data)
                loss = loss_fn(logits, labels)
                return loss

            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
            loss, grads = grad_fn(batch_data, batch_labels)
            optimizer(grads)

            total_loss += loss.asnumpy()

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 5 == 0:
            # Evaluate on test set
            model.set_train(False)
            test_pred = model(test_data)
            test_loss = loss_fn(test_pred, test_labels)

            # Calculate accuracy
            predictions = (test_pred.asnumpy() > 0.5).astype(int)
            accuracy = (predictions == y_test.reshape(-1, 1)).mean()

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.4f}")
            model.set_train()

    # Save model
    model_path = config['models']['risk_uplift_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ms.save_checkpoint(model, model_path)
    logger.info(f"✓ Risk uplift model saved to {model_path}")

    return model


def train_campaign_model(X_train, y_train, X_test, y_test, config):
    """Train Campaign Detection Model (LSTM)"""
    logger.info("=" * 60)
    logger.info("Training Campaign Detection Model (LSTM)")
    logger.info("=" * 60)

    # Create model
    sequence_length = X_train.shape[1]
    input_features = X_train.shape[2]
    model = CampaignDetectionModel(sequence_length=sequence_length, input_features=input_features)

    # Define loss and optimizer
    loss_fn = nn.BCELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config['learning_rate'])

    # Convert to MindSpore tensors
    train_data = Tensor(X_train, ms.float32)
    train_labels = Tensor(y_train.reshape(-1, 1), ms.float32)
    test_data = Tensor(X_test, ms.float32)
    test_labels = Tensor(y_test.reshape(-1, 1), ms.float32)

    # Training loop
    epochs = config['epochs']
    batch_size = config['batch_size']

    model.set_train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(X_train) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            batch_data = train_data[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            # Forward pass
            def forward_fn(data, labels):
                logits = model(data)
                loss = loss_fn(logits, labels)
                return loss

            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
            loss, grads = grad_fn(batch_data, batch_labels)
            optimizer(grads)

            total_loss += loss.asnumpy()

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 5 == 0:
            # Evaluate on test set
            model.set_train(False)
            test_pred = model(test_data)
            test_loss = loss_fn(test_pred, test_labels)

            # Calculate accuracy
            predictions = (test_pred.asnumpy() > 0.5).astype(int)
            accuracy = (predictions == y_test.reshape(-1, 1)).mean()

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.4f}")
            model.set_train()

    # Save model
    model_path = config['models']['campaign_model_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ms.save_checkpoint(model, model_path)
    logger.info(f"✓ Campaign model saved to {model_path}")

    return model


def main():
    """Main training pipeline"""
    logger.info("=" * 70)
    logger.info("PROACTIVE WARNING SERVICE - MODEL TRAINING")
    logger.info("=" * 70)

    # Load configuration
    config_path = '../config/warning_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Training config: {config['training']}")

    # Initialize data processor
    csv_path = '../data/raw/creditcard.csv'
    processor = CreditCardDataProcessor(csv_path)

    # Prepare datasets for each model
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING DATASETS")
    logger.info("=" * 70)

    cohort_data = processor.prepare_cohort_selection_data()
    risk_uplift_data = processor.prepare_risk_uplift_data()
    campaign_data = processor.prepare_campaign_detection_data(sequence_length=24)

    # Train all models
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODELS")
    logger.info("=" * 70)

    cohort_model = train_cohort_model(*cohort_data, config['training'])
    risk_uplift_model = train_risk_uplift_model(*risk_uplift_data, config['training'])
    campaign_model = train_campaign_model(*campaign_data, config['training'])

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"✓ Cohort Selection Model: {config['training']['models']['cohort_model_path']}")
    logger.info(f"✓ Risk Uplift Model: {config['training']['models']['risk_uplift_model_path']}")
    logger.info(f"✓ Campaign Detection Model: {config['training']['models']['campaign_model_path']}")
    logger.info("\nAll models trained successfully!")
    logger.info("You can now run the API: ~/mindspore311_env/bin/python src/api_warning.py")


if __name__ == "__main__":
    main()
