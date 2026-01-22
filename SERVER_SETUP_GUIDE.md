# MAIFDS Server Setup - Simple Guide

## Step-by-Step Deployment (SSH into server and run these commands)

### 1. SSH into your server
```bash
ssh root@62.171.175.58
```

### 2. Create dedicated user for security
```bash
# Create maifds user with home directory
useradd -m -s /bin/bash maifds

# Add to sudo group (optional, for maintenance tasks)
usermod -aG sudo maifds

# Set password for the user
passwd maifds
# (You'll be prompted to enter a password)
```

### 3. Update system and install dependencies
```bash
# Update packages
apt update && apt upgrade -y

# Install Python 3.11 (required by MindSpore)
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install other tools
apt install -y git nginx build-essential

# Configure firewall
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp
ufw --force enable
```

### 4. Setup application directory
```bash
# Create directory
mkdir -p /var/www
cd /var/www

# Clone your repo (update with your actual repo URL)
git clone https://github.com/Michael-cmd-sys/maifds.git
# OR transfer your code using rsync/scp

cd maifds

# Set ownership to maifds user
chown -R maifds:maifds /var/www/maifds
```

### 5. Switch to maifds user and setup Python environment
```bash
# Switch to maifds user
su - maifds
cd /var/www/maifds

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies (this will take 5-10 minutes for MindSpore)
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Build the UI (still as maifds user)
```bash
cd ui
npm install
npm run build
cd ..

# Exit back to root
exit
```

### 7. Create systemd service
```bash
# Create service file (as root)
cat > /etc/systemd/system/maifds.service << 'EOF'
[Unit]
Description=MAIFDS FastAPI Backend
After=network.target

[Service]
Type=simple
User=maifds
Group=maifds
WorkingDirectory=/var/www/maifds
Environment="PATH=/var/www/maifds/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/var/www/maifds/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable maifds
systemctl start maifds

# Check it's running
systemctl status maifds
```

### 8. Configure Nginx
```bash
# Create nginx config (as root)
cat > /etc/nginx/sites-available/maifds << 'EOF'
upstream maifds_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name 62.171.175.58;
    
    client_max_body_size 10M;
    
    # Serve React UI
    location / {
        root /var/www/maifds/ui/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location ~ ^/(v1|health|docs|redoc|openapi.json) {
        proxy_pass http://maifds_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

# Enable site
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/maifds /etc/nginx/sites-enabled/

# Test and restart nginx
nginx -t
systemctl restart nginx
```

### 9. Fix permissions for nginx (if needed)
```bash
# Ensure nginx can read the UI files
chmod -R 755 /var/www/maifds/ui/dist
```

### 10. Test your deployment
```bash
# Test backend
curl http://localhost:8000/health

# Test frontend
curl http://localhost
```

## 🎉 Done! Access your app:
- Frontend: **http://62.171.175.58**
- API Docs: **http://62.171.175.58/docs**

## User Information
- **Application User**: `maifds`
- **Application Directory**: `/var/www/maifds`
- **Running as**: `maifds` user (not root) ✅ More secure!

---

## Useful Commands

### View logs
```bash
# Backend logs
journalctl -u maifds -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Restart services
```bash
systemctl restart maifds
systemctl restart nginx
```

### Update code
```bash
# Switch to maifds user
su - maifds
cd /var/www/maifds
git pull
source .venv/bin/activate
pip install -r requirements.txt
cd ui && npm run build && cd ..
exit

# Restart service (as root)
systemctl restart maifds
```

---

## Optional: Environment Variables

If you need to configure API keys, create a `.env` file:

```bash
cd /var/www/maifds
nano .env
```

Add your configuration (example):
```
APP_ENV=production
MOOLRE_API_KEY=your_key_here
SECRET_KEY=your_secret_key_here
```

---

## SSH Access to maifds User

You can also SSH directly as the maifds user:

```bash
# From your local machine
ssh maifds@62.171.175.58

# Or copy your SSH key for passwordless login
ssh-copy-id maifds@62.171.175.58
```

---

**Need help?** Check logs with `journalctl -u maifds -f` or `nginx -t`
