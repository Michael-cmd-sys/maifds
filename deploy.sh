#!/bin/bash
#
# MAIFDS Deployment Script
# Run this on your Contabo server after running deployment_config.sh
#
# Usage: bash deploy.sh
#

set -e  # Exit on error

DEPLOY_DIR="/var/www/maifds"
REPO_URL="https://github.com/Michael-cmd-sys/maifds.git"  # Update with your repo URL
BRANCH="main"  # Update if needed

echo "========================================="
echo "MAIFDS Deployment"
echo "========================================="

# Step 1: Clone or update repository
if [ -d "$DEPLOY_DIR" ]; then
    echo "[1/8] Updating repository..."
    cd "$DEPLOY_DIR"
    git pull origin "$BRANCH"
else
    echo "[1/8] Cloning repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$DEPLOY_DIR"
    cd "$DEPLOY_DIR"
fi

# Step 2: Create Python virtual environment
echo "[2/8] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Step 3: Upgrade pip and install dependencies
echo "[3/8] Installing Python dependencies (including MindSpore)..."
pip install --upgrade pip
pip install -r requirements.txt

# This may take a while on first install (MindSpore is large)
echo "   ⏳ MindSpore installation may take 5-10 minutes..."

# Step 4: Install UI dependencies
echo "[4/8] Installing UI dependencies..."
cd ui
npm install

# Step 5: Build React UI for production
echo "[5/8] Building React UI for production..."
npm run build

# Verify dist folder was created
if [ ! -d "dist" ]; then
    echo "❌ Error: UI build failed - dist folder not found"
    exit 1
fi

cd ..

# Step 6: Install systemd service
echo "[6/8] Installing systemd service..."
cp maifds.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable maifds
systemctl restart maifds

# Wait for service to start
sleep 3

# Check service status
if systemctl is-active --quiet maifds; then
    echo "   ✅ MAIFDS backend service is running"
else
    echo "   ❌ MAIFDS backend service failed to start"
    journalctl -u maifds -n 50
    exit 1
fi

# Step 7: Configure Nginx
echo "[7/8] Configuring Nginx..."
cp nginx_maifds.conf /etc/nginx/sites-available/maifds

# Remove default site if it exists
if [ -L /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
fi

# Enable MAIFDS site
ln -sf /etc/nginx/sites-available/maifds /etc/nginx/sites-enabled/maifds

# Test nginx configuration
if nginx -t; then
    systemctl restart nginx
    echo "   ✅ Nginx configured and restarted"
else
    echo "   ❌ Nginx configuration test failed"
    exit 1
fi

# Step 8: Verify deployment
echo "[8/8] Verifying deployment..."

# Test backend health endpoint
sleep 2
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ Backend health check passed"
else
    echo "   ⚠️  Backend health check failed (may need more time to start)"
fi

# Test nginx
if curl -f http://localhost > /dev/null 2>&1; then
    echo "   ✅ Frontend serving correctly"
else
    echo "   ⚠️  Frontend serving failed"
fi

echo ""
echo "========================================="
echo "✅ Deployment complete!"
echo "========================================="
echo ""
echo "Your MAIFDS application is now running:"
echo "  - Frontend: http://62.171.175.58"
echo "  - Backend API: http://62.171.175.58/v1/"
echo "  - API Docs: http://62.171.175.58/docs"
echo ""
echo "Useful commands:"
echo "  - View backend logs: journalctl -u maifds -f"
echo "  - Restart backend: systemctl restart maifds"
echo "  - Check backend status: systemctl status maifds"
echo "  - Restart nginx: systemctl restart nginx"
echo "  - Check nginx status: systemctl status nginx"
echo ""
