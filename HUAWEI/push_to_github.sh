#!/bin/bash

# Step 1: Create/Update .gitignore file
cat > .gitignore << 'GITIGNORE'
# Virtual Environments
venv/
mindenv/
env/
ENV/
*.pyc
__pycache__/

# MindSpore specific
*.ckpt
*.meta
*.mindrecord
*.db

# Log files
*.log
phishing_detector_mindspore.log

# Data files (if they're large or sensitive)
data/
*.csv
*.json
!config.json

# IDE specific
.vscode/
.idea/
*.swp
*.swo
*~

# OS specific
.DS_Store
Thumbs.db

# Credentials and sensitive files
*.key
*.pem
.env
secrets.json

# Model checkpoints (can be large)
checkpoints/
models/
*.h5
*.pt
*.pth

# Temporary files
*.tmp
temp/
GITIGNORE

# Step 2: Check current git status
echo "=== Current Git Status ==="
git status

# Step 3: Add the .gitignore file
git add .gitignore

# Step 4: Add all files (respecting .gitignore)
git add .

# Step 5: Check what will be committed
echo ""
echo "=== Files to be committed ==="
git status

# Step 6: Commit the changes
read -p "Enter commit message: " commit_msg
git commit -m "$commit_msg"

# Step 7: Switch to cyril_bot branch (create if doesn't exist)
if git show-ref --verify --quiet refs/heads/cyril_bot; then
    echo "Switching to existing cyril_bot branch..."
    git checkout cyril_bot
else
    echo "Creating new cyril_bot branch..."
    git checkout -b cyril_bot
fi

# Step 8: Push to remote cyril_bot branch
echo ""
echo "=== Pushing to remote cyril_bot branch ==="
git push -u origin cyril_bot

echo ""
echo "=== Push completed! ==="
echo "Your changes are now on the cyril_bot branch"
