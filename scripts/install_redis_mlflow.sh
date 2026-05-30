#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Install Redis + MLflow on EC2 (Ubuntu 22.04 / Amazon Linux 2023)
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
    sudo yum install -y redis6 || sudo amazon-linux-extras install redis6 -y || sudo yum install -y redis
elif [[ "$OS" == "rhel" || "$OS" == "centos" ]]; then
    sudo yum install -y epel-release
    sudo yum install -y redis
fi

# ── FIX 1: Correctly locate redis.conf on Ubuntu 22.04 ───────────────────────
# Ubuntu 22.04 uses /etc/redis/redis.conf
# Amazon Linux / CentOS use /etc/redis.conf
# Try both paths; also handle the protected-mode line that blocks remote connections
REDIS_CONF=""
if [ -f /etc/redis/redis.conf ]; then
    REDIS_CONF="/etc/redis/redis.conf"
elif [ -f /etc/redis.conf ]; then
    REDIS_CONF="/etc/redis.conf"
fi

if [ -n "$REDIS_CONF" ]; then
    echo "  [redis] Config found at: $REDIS_CONF"
    # Change bind from 127.0.0.1 to 0.0.0.0
    sudo sed -i 's/^bind 127.0.0.1 -::1/bind 0.0.0.0/' "$REDIS_CONF"
    sudo sed -i 's/^bind 127.0.0.1$/bind 0.0.0.0/' "$REDIS_CONF"
    # Disable protected-mode so external connections are allowed
    sudo sed -i 's/^protected-mode yes/protected-mode no/' "$REDIS_CONF"
    echo "  [redis] bind set to 0.0.0.0, protected-mode disabled"
else
    echo "  ⚠ redis.conf not found — skipping bind config"
fi

# ── FIX 2: Correct service name on Ubuntu 22.04 is redis-server ──────────────
if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    REDIS_SERVICE="redis-server"
else
    REDIS_SERVICE="redis"
fi

sudo systemctl enable "$REDIS_SERVICE"
sudo systemctl restart "$REDIS_SERVICE"

sleep 2
if redis-cli ping | grep -q "PONG"; then
    echo "  ✓ Redis installed and running on port 6379"
else
    echo "  ✗ Redis failed to start. Check: sudo systemctl status $REDIS_SERVICE"
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  3. INSTALL MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/6] Installing MLflow"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── FIX 3: Export PATH *before* pip install so 'which mlflow' works after ─────
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q '.local/bin' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

pip3 install --user mlflow==2.13.0 boto3

# Verify (PATH is already exported above)
if mlflow --version; then
    echo "  ✓ MLflow installed: $(mlflow --version)"
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

sudo mkdir -p /opt/mlflow/db /opt/mlflow/artifacts
sudo chown -R "$USER":"$USER" /opt/mlflow
echo "  ✓ MLflow directories created at /opt/mlflow/"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  5. CREATE SYSTEMD SERVICE FOR MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/6] Creating MLflow systemd service"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── FIX 4: Resolve MLFLOW_BIN after PATH is exported ─────────────────────────
MLFLOW_BIN="$HOME/.local/bin/mlflow"
if [ ! -f "$MLFLOW_BIN" ]; then
    MLFLOW_BIN=$(which mlflow)
fi
echo "  [mlflow] Binary path: $MLFLOW_BIN"

# ── FIX 5: sqlite URI needs 4 slashes for absolute path (/opt/...) ───────────
# sqlite:////opt/mlflow/db/mlflow.db  ← correct (4 slashes = absolute path)
# sqlite:///opt/mlflow/db/mlflow.db   ← wrong   (3 slashes = relative path)
sudo tee /etc/systemd/system/mlflow.service > /dev/null <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/mlflow
ExecStart=$MLFLOW_BIN server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////opt/mlflow/db/mlflow.db \
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

# Wait longer — MLflow takes a few seconds to initialise the SQLite DB
sleep 6

if curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200"; then
    echo "  ✓ MLflow running on port 5000"
else
    echo "  ⚠ MLflow may still be starting. Checking status..."
    sudo systemctl status mlflow --no-pager || true
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
echo "  │ 5000     │ TCP      │ MLflow UI                   │"
echo "  │ 8000     │ TCP      │ FastAPI                     │"
echo "  │ 6379     │ TCP      │ Redis (restrict to your IP) │"
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
echo "    Redis  → redis://localhost:6379"
echo "    MLflow → http://$PUBLIC_IP:5000"
echo ""
echo "  Verify:"
echo "    redis-cli ping                         # PONG"
echo "    sudo systemctl status $REDIS_SERVICE"
echo "    sudo systemctl status mlflow"
echo "    sudo journalctl -u mlflow -f           # live logs"
echo ""
echo "  .env values:"
echo "    REDIS_HOST=localhost"
echo "    REDIS_PORT=6379"
echo "    MLFLOW_TRACKING_URI=http://localhost:5000"
echo ""
echo "═══════════════════════════════════════════════════════"
