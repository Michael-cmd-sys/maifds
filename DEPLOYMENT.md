# MAIFDS Server Deployment Guide

This directory contains all files needed to deploy MAIFDS to a production server.

## 📋 Deployment Files

| File | Purpose |
|------|---------|
| `deployment_config.sh` | Initial server setup (install Python, Node.js, Nginx) |
| `deploy.sh` | Application deployment script |
| `maifds.service` | Systemd service configuration for backend |
| `nginx_maifds.conf` | Nginx reverse proxy configuration |
| `.env.production` | Environment variables template |

## 🚀 Quick Deployment Steps

### Step 1: Prepare Your Server

SSH into your Contabo server:
```bash
ssh root@62.171.175.58
```

### Step 2: Run Server Setup

On your **local machine**, transfer the setup script:
```bash
scp deployment_config.sh root@62.171.175.58:/root/
```

On the **server**:
```bash
bash /root/deployment_config.sh
```

⏱️ This will take 5-10 minutes to install all dependencies.

### Step 3: Deploy Application

**Option A: Deploy from GitHub** (Recommended)

1. Push your code to GitHub
2. Update `deploy.sh` line 10 with your repository URL:
   ```bash
   REPO_URL="https://github.com/YOUR_USERNAME/maifds.git"
   ```
3. Transfer deployment files to server:
   ```bash
   scp deploy.sh maifds.service nginx_maifds.conf .env.production root@62.171.175.58:/root/
   ```
4. On the server:
   ```bash
   bash /root/deploy.sh
   ```

**Option B: Deploy via rsync** (Alternative)

Transfer entire project from local machine:
```bash
rsync -avz --exclude 'node_modules' --exclude '.venv' --exclude 'ui/dist' \
  /home/senanu/Desktop/work/maifds/ root@62.171.175.58:/var/www/maifds/
```

Then on the server:
```bash
cd /var/www/maifds
bash deploy.sh
```

### Step 4: Configure Environment Variables

On the server:
```bash
cd /var/www/maifds
cp .env.production .env
nano .env  # Edit with your actual API keys and configuration
```

### Step 5: Verify Deployment

Test the endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Check frontend
curl http://localhost

# Check services
systemctl status maifds
systemctl status nginx
```

## 🌐 Access Your Application

- **Frontend**: http://62.171.175.58
- **API Documentation**: http://62.171.175.58/docs
- **Health Check**: http://62.171.175.58/health

## 📝 Management Commands

### Backend Service
```bash
# View logs
journalctl -u maifds -f

# Restart service
systemctl restart maifds

# Stop service
systemctl stop maifds

# Check status
systemctl status maifds
```

### Nginx
```bash
# View access logs
tail -f /var/log/nginx/maifds_access.log

# View error logs
tail -f /var/log/nginx/maifds_error.log

# Test configuration
nginx -t

# Restart
systemctl restart nginx
```

### Update Application
```bash
cd /var/www/maifds
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt

# Rebuild UI
cd ui && npm run build && cd ..

# Restart services
systemctl restart maifds
```

## 🔒 Security Enhancements (Optional)

### 1. Set Up SSL/HTTPS with Let's Encrypt

```bash
# Install certbot
apt install -y certbot python3-certbot-nginx

# Get certificate (replace with your domain)
certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
```

### 2. Create Dedicated User (Instead of Running as Root)

```bash
# Create maifds user
useradd -m -s /bin/bash maifds

# Transfer ownership
chown -R maifds:maifds /var/www/maifds

# Update maifds.service to use: User=maifds
```

### 3. Set Up Firewall Rules

Already configured in `deployment_config.sh`, but you can verify:
```bash
ufw status
```

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check logs
journalctl -u maifds -n 100

# Common issues:
# - Python dependencies missing: rerun pip install -r requirements.txt
# - Port 8000 in use: lsof -i :8000
# - Virtual environment not activated in service file
```

### Frontend shows 404 or blank page
```bash
# Verify dist folder exists
ls -la /var/www/maifds/ui/dist

# Rebuild if needed
cd /var/www/maifds/ui && npm run build

# Check nginx config
nginx -t
tail -f /var/log/nginx/maifds_error.log
```

### API requests fail from frontend
- Check CORS settings in `app.py`
- Verify nginx is proxying `/v1/*` correctly
- Check backend logs: `journalctl -u maifds -f`

## 📊 Monitoring

### System Resources
```bash
# CPU and memory usage
htop

# Disk usage
df -h

# Backend process
ps aux | grep uvicorn
```

### Application Health
```bash
# Quick health check
curl http://localhost:8000/health

# Detailed API test
curl -X POST http://localhost:8000/v1/blacklist/check \
  -H "Content-Type: application/json" \
  -d '{"phone_number":"+233501234567"}'
```

## 🔄 Backup Strategy

### Database Backup
```bash
# Backup SQLite databases
tar -czf maifds_backup_$(date +%Y%m%d).tar.gz \
  /var/www/maifds/customer_reputation_system/data/database

# Copy to safe location or remote server
scp maifds_backup_*.tar.gz user@backup-server:/backups/
```

---

**Need help?** Check the logs first, then review the implementation plan or contact the development team.
