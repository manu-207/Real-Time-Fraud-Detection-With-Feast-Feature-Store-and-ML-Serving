#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Install Redis + MLflow on EC2 (Ubuntu 22.04-25.x / Amazon Linux 2023)
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
echo "[setup] Python: $(python3 --version)"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  1. SYSTEM UPDATE
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/6] Updating system packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    sudo apt-get update -y
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

# ── Fix redis.conf bind address ───────────────────────────────────────────────
# Ubuntu 22.04+ uses /etc/redis/redis.conf
# Amazon Linux / CentOS use /etc/redis.conf
REDIS_CONF=""
if [ -f /etc/redis/redis.conf ]; then
    REDIS_CONF="/etc/redis/redis.conf"
elif [ -f /etc/redis.conf ]; then
    REDIS_CONF="/etc/redis.conf"
fi

if [ -n "$REDIS_CONF" ]; then
    echo "  [redis] Config: $REDIS_CONF"
    # Handle both "bind 127.0.0.1 -::1" (Ubuntu 22+) and "bind 127.0.0.1" forms
    sudo sed -i 's/^bind 127\.0\.0\.1.*/bind 0.0.0.0/' "$REDIS_CONF"
    sudo sed -i 's/^protected-mode yes/protected-mode no/' "$REDIS_CONF"
    echo "  [redis] bind → 0.0.0.0, protected-mode → no"
fi

# ── FIX: Ubuntu 25 ships redis.service as a symlink which blocks 'enable' ────
# Use redis-server.service directly; suppress the linked-unit error on redis.service
if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    REDIS_SERVICE="redis-server"
else
    REDIS_SERVICE="redis"
fi

sudo systemctl enable "$REDIS_SERVICE" 2>/dev/null || true
sudo systemctl restart "$REDIS_SERVICE"

sleep 2
if redis-cli ping | grep -q "PONG"; then
    echo "  ✓ Redis running on port 6379"
else
    echo "  ✗ Redis failed to start"
    sudo systemctl status "$REDIS_SERVICE" --no-pager || true
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  3. INSTALL MLFLOW (into a dedicated venv)
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/6] Installing MLflow"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── FIX: Python 3.12+ (PEP 668) blocks pip3 --user on Ubuntu 24/25 ───────────
# Solution: install MLflow into a dedicated venv at /opt/mlflow-venv
# This is cleaner than --break-system-packages and survives OS upgrades.
MLFLOW_VENV="/opt/mlflow-venv"

if [ ! -d "$MLFLOW_VENV" ]; then
    echo "  [mlflow] Creating venv at $MLFLOW_VENV"
    sudo python3 -m venv "$MLFLOW_VENV"
    sudo chown -R "$USER":"$USER" "$MLFLOW_VENV"
fi

"$MLFLOW_VENV/bin/pip" install --upgrade pip --quiet
"$MLFLOW_VENV/bin/pip" install mlflow==2.13.0 boto3 --quiet

MLFLOW_BIN="$MLFLOW_VENV/bin/mlflow"

if "$MLFLOW_BIN" --version; then
    echo "  ✓ MLflow installed in $MLFLOW_VENV"
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
echo "  ✓ MLflow directories at /opt/mlflow/"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  5. CREATE SYSTEMD SERVICE FOR MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/6] Creating MLflow systemd service"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  [mlflow] Binary: $MLFLOW_BIN"

# sqlite:////opt/... — 4 slashes required for absolute path
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
Environment="PATH=$MLFLOW_VENV/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow

# MLflow takes a few seconds to initialise the SQLite DB on first start
echo "  [mlflow] Waiting for MLflow to start..."
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 | grep -q "200"; then
        echo "  ✓ MLflow running on port 5000"
        break
    fi
    sleep 2
    if [ "$i" -eq 10 ]; then
        echo "  ⚠ MLflow didn't respond in 20s — checking logs:"
        sudo journalctl -u mlflow -n 20 --no-pager || true
    fi
done
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
echo "  ┌──────────┬──────────┬─────────────────────────────────┐"
echo "  │ Port     │ Protocol │ Purpose                         │"
echo "  ├──────────┼──────────┼─────────────────────────────────┤"
echo "  │ 5000     │ TCP      │ MLflow UI                       │"
echo "  │ 8000     │ TCP      │ FastAPI                         │"
echo "  │ 6379     │ TCP      │ Redis (restrict to your IP!)    │"
echo "  └──────────┴──────────┴─────────────────────────────────┘"
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
echo "    Redis  → localhost:6379"
echo "    MLflow → http://$PUBLIC_IP:5000"
echo ""
echo "  Verify:"
echo "    redis-cli ping"
echo "    sudo systemctl status $REDIS_SERVICE"
echo "    sudo systemctl status mlflow"
echo "    sudo journalctl -u mlflow -f"
echo ""
echo "  .env values:"
echo "    REDIS_HOST=localhost"
echo "    REDIS_PORT=6379"
echo "    MLFLOW_TRACKING_URI=http://localhost:5000"
echo ""
echo "═══════════════════════════════════════════════════════"
