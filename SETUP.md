# MAIFDS - Quick Setup Guide

**One command to rule them all** - Get the entire project running in minutes, not hours.

## ⚠️ Important: Python Version Requirement

**MindSpore requires Python 3.9, 3.10, or 3.11 only!**

The setup script will automatically try to use Python 3.11 (or 3.10/3.9 if 3.11 is not available).

If you don't have a compatible Python version, install it first:
```bash
# Using pyenv (recommended)
pyenv install 3.11.9
pyenv local 3.11.9

# Or using your system package manager
# Ubuntu/Debian:
sudo apt install python3.11 python3.11-venv

# macOS (with Homebrew):
brew install python@3.11
```

## 🚀 Quick Start (Recommended)

### Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Python 3.9, 3.10, or 3.11** (MindSpore requirement)

### Setup (One Command)

```bash
# Linux/macOS
./setup.sh

# Windows (PowerShell)
.\setup.ps1
```

That's it! 🎉 The script will:
- ✅ Create a virtual environment with compatible Python version
- ✅ Install all dependencies
- ✅ Set up the project

## 📋 Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# 1. Create virtual environment with Python 3.11 (or 3.10/3.9)
uv venv .venv --python 3.11

# 2. Activate it
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\Activate.ps1  # Windows

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Install project in editable mode
uv pip install -e .
```

## 🎯 Using the Environment

### Activate the environment:
```bash
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows
```

### Or use uv directly (no activation needed):
```bash
uv run python customer-reputation-system/main.py
uv run python mel_dev/features/call_triggered_defense/src/train.py
```

### Deactivate:
```bash
deactivate
```

## 📦 What Gets Installed?

The centralized `requirements.txt` includes all dependencies for:

- ✅ **Customer Reputation System** (with NLP Feature 2)
- ✅ **MEL Dev Features** (Call Triggered Defense, Click-TX Correlation, etc.)
- ✅ **HUAWEI Services** (Phishing Detector, Blacklist Service, Proactive Warning)
- ✅ **MindSpore** (AI framework) - Version 2.4.1+ (Python 3.9-3.11 only)
- ✅ **All supporting libraries** (pandas, numpy, Flask, etc.)

## 🔧 MindSpore Installation

By default, the setup installs **MindSpore CPU version 2.4.1+** (good for development).

### For GPU Support:
1. Edit `requirements.txt`
2. Comment out: `mindspore>=2.4.1`
3. Uncomment: `mindspore-gpu>=2.2.14`
4. Re-run: `uv pip install -r requirements.txt`

### For Ascend Hardware:
1. Follow [MindSpore installation guide](https://www.mindspore.cn/install/en)
2. Install MindSpore separately based on your hardware

## 🧪 Verify Installation

```bash
# Check Python version (should be 3.9, 3.10, or 3.11)
python --version

# Check MindSpore
python -c "import mindspore; print(f'MindSpore {mindspore.__version__}')"

# Run tests
pytest customer-reputation-system/tests/
```

## 📁 Project Structure

```
maifds/
├── requirements.txt          # ← Centralized dependencies (use this!)
├── setup.sh                 # ← One-command setup (Linux/macOS)
├── setup.ps1                # ← One-command setup (Windows)
├── pyproject.toml           # ← Project metadata
├── customer-reputation-system/
│   └── requirements.txt     # (deprecated - use root requirements.txt)
├── mel_dev/
│   └── requirements.txt     # (deprecated - use root requirements.txt)
└── HUAWEI/
    └── requirements.txt     # (deprecated - use root requirements.txt)
```

## ❓ Troubleshooting

### "uv: command not found"
- Install uv using the commands above
- Make sure it's in your PATH

### "Python 3.13 not supported" or "No solution found"
- **MindSpore only supports Python 3.9, 3.10, and 3.11**
- Install Python 3.11: `pyenv install 3.11.9` or use your system package manager
- The setup script will try to use Python 3.11 automatically

### "Permission denied" on setup.sh
```bash
chmod +x setup.sh
```

### MindSpore installation fails
- **Check your Python version** (must be 3.9, 3.10, or 3.11)
- For GPU: Ensure CUDA is installed
- Visit [MindSpore installation guide](https://www.mindspore.cn/install/en)

### Port conflicts
- Check if ports 5000, 8000 are in use
- Modify Flask/API ports in config files

### "tool.uv.dev-dependencies is deprecated"
- This is fixed in the latest version
- Update your `pyproject.toml` if you see this warning

## 🎓 Next Steps

After setup, explore:

1. **Customer Reputation System:**
   ```bash
   cd customer-reputation-system
   python main.py
   ```

2. **Train NLP Model:**
   ```bash
   cd customer-reputation-system/src/nlp
   python train.py
   ```

3. **MEL Dev Features:**
   ```bash
   cd mel_dev/features/call_triggered_defense/src
   python train.py
   ```

## 📚 Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [MindSpore Documentation](https://www.mindspore.cn/)
- [MindSpore Installation Guide](https://www.mindspore.cn/install/en)
- Project README: See `README.md`

---

**Need help?** Check the project README or open an issue on GitHub.
