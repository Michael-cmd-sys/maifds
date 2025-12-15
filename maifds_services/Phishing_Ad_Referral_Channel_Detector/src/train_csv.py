"""
Training script for MindSpore-based Phishing Detector using CSV dataset
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mindspore
from mindspore import nn, Tensor, context
from mindspore.train import Model, Callback, LossMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn.metrics import Accuracy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MindSpore context
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class PhishingNet(nn.Cell):
    """Neural network for phishing detection"""
    
    def __init__(self, input_size=30, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(PhishingNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Dense(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=1-dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Dense(prev_size, 2))  # Binary classification
        
        self.network = nn.SequentialCell(layers)
    
    def construct(self, x):
        return self.network(x)


class TrainingCallback(Callback):
    """Custom callback for training progress"""
    
    def __init__(self, eval_interval=5):
        super(TrainingCallback, self).__init__()
        self.eval_interval = eval_interval
        self.epoch_num = 0
    
    def on_train_epoch_end(self, run_context):
        """Called at the end of each epoch"""
        self.epoch_num += 1
        cb_params = run_context.original_args()
        
        if self.epoch_num % self.eval_interval == 0:
            logger.info(f"Epoch {self.epoch_num} completed")


def load_dataset(csv_path):
    """Load and preprocess the phishing dataset"""
    logger.info(f"Loading dataset from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Separate features and labels
    # The last column is typically the label (Result)
    X = df.iloc[:, 1:-1].values  # Skip index column, take all feature columns
    y = df.iloc[:, -1].values     # Last column is the label
    
    # Convert labels: -1 -> 0 (legitimate), 1 -> 1 (phishing)
    y = np.where(y == -1, 0, 1)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Unique labels: {np.unique(y)}")
    logger.info(f"Class distribution: Legit={np.sum(y==0)}, Phishing={np.sum(y==1)}")
    
    return X, y


def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split and normalize the data"""
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Convert to MindSpore tensors
    X_train_tensor = Tensor(X_train, mindspore.float32)
    y_train_tensor = Tensor(y_train, mindspore.int32)
    X_test_tensor = Tensor(X_test, mindspore.float32)
    y_test_tensor = Tensor(y_test, mindspore.int32)
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor), scaler


def train_model(X_train, y_train, X_test, y_test, config):
    """Train the phishing detection model"""
    
    input_size = X_train.shape[1]
    logger.info(f"Input size: {input_size}")
    
    # Create model
    net = PhishingNet(
        input_size=input_size,
        hidden_sizes=config['model']['hidden_sizes'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Define loss and optimizer
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=config['model']['learning_rate'])
    
    # Create model
    model = Model(net, loss_fn, optimizer, metrics={'accuracy': Accuracy()})
    
    # Setup callbacks
    config_ck = CheckpointConfig(
        save_checkpoint_steps=100,
        keep_checkpoint_max=5
    )
    
    models_dir = config['storage']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    ckpoint_cb = ModelCheckpoint(
        prefix=config['storage']['checkpoint_prefix'],
        directory=models_dir,
        config=config_ck
    )
    
    train_callback = TrainingCallback(eval_interval=10)
    loss_cb = LossMonitor(per_print_times=10)
    
    # Train
    logger.info("Starting training...")
    epochs = config['model']['epochs']
    
    # Create dataset
    from mindspore.dataset import NumpySlicesDataset
    
    train_dataset = NumpySlicesDataset(
        (X_train.asnumpy(), y_train.asnumpy()),
        column_names=['features', 'labels'],
        shuffle=True
    )
    train_dataset = train_dataset.batch(config['model']['batch_size'])
    
    # Train the model
    model.train(
        epochs,
        train_dataset,
        callbacks=[train_callback, loss_cb, ckpoint_cb],
        dataset_sink_mode=False
    )
    
    logger.info("Training completed!")
    
    # Evaluate on test set
    test_dataset = NumpySlicesDataset(
        (X_test.asnumpy(), y_test.asnumpy()),
        column_names=['features', 'labels'],
        shuffle=False
    )
    test_dataset = test_dataset.batch(config['model']['batch_size'])
    
    logger.info("\nEvaluating on test set...")
    result = model.eval(test_dataset, dataset_sink_mode=False)
    logger.info(f"Test accuracy: {result['accuracy']:.4f}")
    
    return model, net


def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("MindSpore Phishing Detector - CSV Training Script")
    logger.info("=" * 70)
    
    # Load configuration
    import json
    # Get the script's directory for all relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir, "..", "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'model': {
                'hidden_sizes': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32
            },
            'storage': {
                'models_dir': os.path.join(script_dir, "..", "data", "processed", "models"),
                'checkpoint_prefix': 'phishing_detector'
            }
        }

    # Load dataset
    csv_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    X, y = load_dataset(csv_path)
    
    # Prepare data
    logger.info("\nPreparing data...")
    (X_train, y_train, X_test, y_test), scaler = prepare_data(X, y)
    
    # Train model
    logger.info("\n" + "=" * 70)
    logger.info("Training Neural Network...")
    logger.info("=" * 70)
    
    try:
        model, net = train_model(X_train, y_train, X_test, y_test, config)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Training Complete!")
        logger.info("=" * 70)
        
        # Save scaler
        import pickle
        scaler_path = os.path.join(config['storage']['models_dir'], 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"\nScaler saved to: {scaler_path}")
        logger.info(f"Model checkpoints saved to: {config['storage']['models_dir']}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
