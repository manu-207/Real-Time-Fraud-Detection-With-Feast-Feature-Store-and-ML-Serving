#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Install Redis on EC2 (Ubuntu 22.04-25.x / Amazon Linux 2023)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    chmod +x scripts/install_redis.sh
#    ./scripts/install_redis.sh
#
#  After running:
#    Redis → localhost:6379
#
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════"
echo "  EC2 Setup: Redis"
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
echo "  [1/3] Updating system packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    sudo apt-get update -y
    sudo apt-get install -y curl wget
elif [[ "$OS" == "amzn" || "$OS" == "rhel" || "$OS" == "centos" ]]; then
    sudo yum update -y
    sudo yum install -y curl wget
fi
echo "  ✓ System updated"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  2. INSTALL REDIS
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/3] Installing Redis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    sudo apt-get install -y redis-server
elif [[ "$OS" == "amzn" ]]; then
    sudo yum install -y redis6 || sudo amazon-linux-extras install redis6 -y || sudo yum install -y redis
elif [[ "$OS" == "rhel" || "$OS" == "centos" ]]; then
    sudo yum install -y epel-release
    sudo yum install -y redis
fi

# ── Fix redis.conf: bind address + protected-mode ────────────────────────────
# Ubuntu 22.04+ → /etc/redis/redis.conf
# Amazon Linux / CentOS → /etc/redis.conf
REDIS_CONF=""
if [ -f /etc/redis/redis.conf ]; then
    REDIS_CONF="/etc/redis/redis.conf"
elif [ -f /etc/redis.conf ]; then
    REDIS_CONF="/etc/redis.conf"
fi

if [ -n "$REDIS_CONF" ]; then
    echo "  [redis] Config found at: $REDIS_CONF"
    # Ubuntu 22+ uses "bind 127.0.0.1 -::1", older uses "bind 127.0.0.1"
    sudo sed -i 's/^bind 127\.0\.0\.1.*/bind 0.0.0.0/' "$REDIS_CONF"
    sudo sed -i 's/^protected-mode yes/protected-mode no/' "$REDIS_CONF"
    echo "  [redis] bind → 0.0.0.0"
    echo "  [redis] protected-mode → no"
else
    echo "  ⚠ redis.conf not found — skipping bind config"
fi

# ── Enable and start ──────────────────────────────────────────────────────────
# Ubuntu/Debian: service is redis-server
# redis.service on Ubuntu 25 is a symlink — suppress the linked-unit error
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
    echo "  ✗ Redis failed to start. Logs:"
    sudo systemctl status "$REDIS_SERVICE" --no-pager || true
    exit 1
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  3. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/3] Done"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Redis Installation Complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Redis → localhost:6379"
echo ""
echo "  Useful commands:"
echo "    redis-cli ping                          # PONG"
echo "    sudo systemctl status $REDIS_SERVICE"
echo "    sudo systemctl restart $REDIS_SERVICE"
echo "    redis-cli info server                   # full info"
echo ""
echo "  .env value:"
echo "    REDIS_HOST=localhost"
echo "    REDIS_PORT=6379"
echo ""
echo "═══════════════════════════════════════════════════════"
