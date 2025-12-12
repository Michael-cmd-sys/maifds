# Quick Start Guide

## ✅ Setup Complete? Test It!

Your virtual environment is ready. Here's how to use it:

### Option 1: Direct Python (Recommended for Fish Shell)

```bash
# Use the Python from the virtual environment directly
.venv/bin/python customer-reputation-system/main.py

# Or activate first (bash/zsh)
source .venv/bin/activate
python customer-reputation-system/main.py
```

### Option 2: Fish Shell Activation

If you're using Fish shell, use the fish-specific activation:

```bash
source .venv/bin/activate.fish
python customer-reputation-system/main.py
```

### Option 3: Use uv (if not using optional dependencies)

```bash
# Note: uv run may try to resolve pyproject.toml dependencies
# For this monorepo, use direct Python instead
.venv/bin/python customer-reputation-system/main.py
```

## 🧪 Verify Installation

```bash
# Check Python version (should be 3.11)
.venv/bin/python --version

# Check MindSpore
.venv/bin/python -c "import mindspore; print(f'MindSpore {mindspore.__version__}')"

# Run a test
.venv/bin/python customer-reputation-system/main.py
```

## 📝 Common Commands

```bash
# Run Customer Reputation System demo
.venv/bin/python customer-reputation-system/main.py

# Train NLP model
.venv/bin/python customer-reputation-system/src/nlp/train.py

# Test NLP inference
.venv/bin/python customer-reputation-system/src/nlp/test_inference.py

# Run MEL Dev features
.venv/bin/python mel_dev/features/call_triggered_defense/src/train.py
```

## 🔧 Troubleshooting

### Fish Shell Issues
If you see activation errors with Fish shell, just use the Python directly:
```bash
.venv/bin/python <script.py>
```

### uv run Issues
If `uv run` gives dependency errors, use direct Python instead:
```bash
.venv/bin/python <script.py>
```

The virtual environment has everything installed - you don't need `uv run` for this monorepo.

