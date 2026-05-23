#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Install Redis + MLflow on EC2 (Ubuntu 22.04 / Amazon Linux 2023)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    chmod +x scripts/install_redis_mlflow.sh
#    ./scripts/install_redis_mlflow.sh
#
#  After running:
#    Redis  → localhost:6379
#    MLflow → http://<ec2-public-ip>:5000
#
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════"
echo "  EC2 Setup: Redis + MLflow"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Detect OS ─────────────────────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi
echo "[setup] Detected OS: $OS"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  1. SYSTEM UPDATE
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/6] Updating system packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    sudo apt-get update -y
    sudo apt-get upgrade -y
    sudo apt-get install -y python3 python3-pip python3-venv gcc make curl wget
elif [[ "$OS" == "amzn" || "$OS" == "rhel" || "$OS" == "centos" ]]; then
    sudo yum update -y
    sudo yum install -y python3 python3-pip gcc make curl wget
fi
echo "  ✓ System updated"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  2. INSTALL REDIS
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/6] Installing Redis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    sudo apt-get install -y redis-server
elif [[ "$OS" == "amzn" ]]; then
    # Amazon Linux 2023
    sudo yum install -y redis6 || sudo amazon-linux-extras install redis6 -y || sudo yum install -y redis
elif [[ "$OS" == "rhel" || "$OS" == "centos" ]]; then
    sudo yum install -y epel-release
    sudo yum install -y redis
fi

# Configure Redis to accept connections from the API container/app
sudo sed -i 's/^bind 127.0.0.1.*/bind 0.0.0.0/' /etc/redis/redis.conf 2>/dev/null || \
sudo sed -i 's/^bind 127.0.0.1.*/bind 0.0.0.0/' /etc/redis.conf 2>/dev/null || true

# Enable and start Redis
sudo systemctl enable redis || sudo systemctl enable redis-server
sudo systemctl start redis || sudo systemctl start redis-server

# Verify
sleep 2
if redis-cli ping | grep -q "PONG"; then
    echo "  ✓ Redis installed and running on port 6379"
else
    echo "  ✗ Redis failed to start. Check: sudo systemctl status redis"
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  3. INSTALL MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/6] Installing MLflow"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install MLflow with pip
pip3 install --user mlflow==2.13.0 boto3

# Add local bin to PATH if not already there
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q '.local/bin' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Verify
if mlflow --version; then
    echo "  ✓ MLflow installed"
else
    echo "  ✗ MLflow installation failed"
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  4. CREATE MLFLOW DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/6] Setting up MLflow storage"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sudo mkdir -p /opt/mlflow/{db,artifacts}
sudo chown -R $USER:$USER /opt/mlflow
echo "  ✓ MLflow directories created at /opt/mlflow/"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  5. CREATE SYSTEMD SERVICE FOR MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/6] Creating MLflow systemd service"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MLFLOW_BIN=$(which mlflow || echo "$HOME/.local/bin/mlflow")

sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/mlflow
ExecStart=$MLFLOW_BIN server \\
    --host 0.0.0.0 \\
    --port 5000 \\
    --backend-store-uri sqlite:///opt/mlflow/db/mlflow.db \\
    --default-artifact-root /opt/mlflow/artifacts
Restart=always
RestartSec=5
Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow

# Wait for MLflow to start
sleep 3
if curl -s http://localhost:5000/health | grep -q "OK" 2>/dev/null || curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200"; then
    echo "  ✓ MLflow running on port 5000"
else
    echo "  ⚠ MLflow may still be starting. Check: sudo systemctl status mlflow"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  6. FIREWALL / SECURITY GROUP REMINDER
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [6/6] Security Group Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "  ⚠ IMPORTANT: Open these ports in your EC2 Security Group:"
echo ""
echo "  ┌──────────┬──────────┬─────────────────────────────┐"
echo "  │ Port     │ Protocol │ Purpose                     │"
echo "  ├──────────┼──────────┼─────────────────────────────┤"
echo "  │ 6379     │ TCP      │ Redis                       │"
echo "  │ 5000     │ TCP      │ MLflow UI                   │"
echo "  │ 8000     │ TCP      │ FastAPI (if running here)   │"
echo "  └──────────┴──────────┴─────────────────────────────┘"
echo ""
echo "  AWS Console → EC2 → Security Groups → Inbound Rules → Edit"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<your-ec2-ip>")

echo "═══════════════════════════════════════════════════════"
echo "  ✅ Installation Complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Services:"
echo "    Redis  → redis://$PUBLIC_IP:6379"
echo "    MLflow → http://$PUBLIC_IP:5000"
echo ""
echo "  Useful commands:"
echo "    redis-cli ping                    # test Redis"
echo "    sudo systemctl status redis       # Redis status"
echo "    sudo systemctl status mlflow      # MLflow status"
echo "    sudo journalctl -u mlflow -f      # MLflow logs"
echo ""
echo "  Set in your .env file:"
echo "    REDIS_HOST=$PUBLIC_IP"
echo "    REDIS_PORT=6379"
echo "    MLFLOW_TRACKING_URI=http://$PUBLIC_IP:5000"
echo ""
echo "═══════════════════════════════════════════════════════"
