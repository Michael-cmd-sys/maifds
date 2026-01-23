#!/bin/bash
#
# MAIFDS Server Configuration Script
# Run this on your Contabo Ubuntu VPS as root
#
# Usage: bash deployment_config.sh
#

set -e  # Exit on error

echo "========================================="
echo "MAIFDS Server Configuration"
echo "========================================="

# Update system
echo "[1/7] Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.11 (required by MindSpore)
echo "[2/7] Installing Python 3.11..."
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Make Python 3.11 the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --set python3 /usr/bin/python3.11

# Install Node.js 20.x LTS
echo "[3/7] Installing Node.js 20.x..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install build tools and dependencies
echo "[4/7] Installing build tools..."
apt install -y \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    wget \
    nginx \
    ufw

# Configure firewall
echo "[5/7] Configuring firewall..."
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw --force enable

# Install pip and upgrade
echo "[6/7] Upgrading pip..."
python3 -m pip install --upgrade pip

# Verify installations
echo "[7/7] Verifying installations..."
echo "Python version: $(python3 --version)"
echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"
echo "Nginx version: $(nginx -v 2>&1)"

echo ""
echo "========================================="
echo "✅ Server configuration complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Clone your repository to /var/www/maifds"
echo "2. Run the deploy.sh script"
