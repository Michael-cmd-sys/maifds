#!/bin/bash
# MAIFDS - One-Command Setup Script
# This script sets up the entire project environment using uv

set -e  # Exit on error

echo "=========================================="
echo "  MAIFDS Project Setup"
echo "  AI powered cyber protection for Ghana's MoMo ecosystem"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ uv found: $(uv --version)"
echo ""

# Create virtual environment with uv (using Python 3.11 for MindSpore compatibility)
echo "📦 Creating virtual environment..."
echo "   Note: MindSpore requires Python 3.9-3.11..."

# Try to find and use a compatible Python version
if command -v python3.11 &> /dev/null; then
    echo "   Using Python 3.11..."
    uv venv .venv --python 3.11
elif command -v python3.10 &> /dev/null; then
    echo "   Using Python 3.10..."
    uv venv .venv --python 3.10
elif command -v python3.9 &> /dev/null; then
    echo "   Using Python 3.9..."
    uv venv .venv --python 3.9
else
    echo "   ⚠️  Warning: No Python 3.9-3.11 found, using default Python..."
    echo "   This may fail if your Python version is not compatible with MindSpore."
    echo ""
    echo "   To install Python 3.11:"
    echo "     # Using pyenv (recommended):"
    echo "     pyenv install 3.11.9"
    echo "     pyenv local 3.11.9"
    echo ""
    echo "     # Or using system package manager:"
    echo "     # Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "     # macOS: brew install python@3.11"
    echo ""
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    uv venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
uv pip install -r requirements.txt

# Note: Skipping editable install - this is a monorepo with independent modules
# Each module can be run directly without package installation
echo "✅ Setup complete! You can now run modules directly."

echo ""
echo "=========================================="
echo "  ✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo ""
echo "  Option 1: Direct Python (works everywhere):"
echo "    .venv/bin/python <script.py>"
echo ""
echo "  Option 2: Activate first (bash/zsh):"
echo "    source .venv/bin/activate"
echo "    python <script.py>"
echo ""
echo "  Option 3: Fish shell:"
echo "    source .venv/bin/activate.fish"
echo "    python <script.py>"
echo ""
echo "See QUICKSTART.md for more examples!"
echo ""

