#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Setting up MindSpore Phishing Detector environment..."

# 1. Create a Python virtual environment if it doesn't exist
if [ ! -d "venv_mindspore" ]; then
    echo "Creating virtual environment 'venv_mindspore'..."
    python3 -m venv venv_mindspore
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source venv_mindspore/bin/activate

# 3. Install dependencies from requirements_mindspore.txt
echo "Installing dependencies from requirements_mindspore.txt..."
pip install --upgrade pip
pip install -r requirements_mindspore.txt

# 4. Download spaCy English model
echo "Downloading spaCy English model (en_core_web_sm)..."
python -m spacy download en_core_web_sm

echo "MindSpore Phishing Detector environment setup complete."
echo "To activate the environment, run: source venv_mindspore/bin/activate"
