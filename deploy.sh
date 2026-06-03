#!/bin/bash
# deploy.sh — one-shot deployment script for Ubuntu/Debian VPS
# Usage: bash deploy.sh
set -e

REPO="digiservbiz/Crypto-Trading-Bot"
BRANCH="claude/setup-crypto-trading-bot-i3OqL"
APP_DIR="$HOME/crypto-trading-bot"

echo "======================================================"
echo " Crypto Trading Bot — VPS Deployment"
echo "======================================================"

# ── 1. System packages ────────────────────────────────────
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git curl

# ── 2. Docker ─────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "[2/6] Installing Docker..."
  curl -fsSL https://get.docker.com | sudo bash
  sudo usermod -aG docker "$USER"
  echo "      Docker installed. NOTE: log out and back in if this is your first install."
else
  echo "[2/6] Docker already installed ($(docker --version))"
fi

# Docker Compose v2 (plugin)
if ! docker compose version &>/dev/null 2>&1; then
  echo "      Installing Docker Compose plugin..."
  sudo apt-get install -y -qq docker-compose-plugin
fi

# ── 3. Clone / update repo ────────────────────────────────
echo "[3/6] Cloning repository..."
if [ -d "$APP_DIR/.git" ]; then
  echo "      Repo already cloned — pulling latest..."
  git -C "$APP_DIR" fetch origin
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull origin "$BRANCH"
else
  git clone --branch "$BRANCH" "https://github.com/$REPO.git" "$APP_DIR"
fi

cd "$APP_DIR"

# ── 4. Environment file ───────────────────────────────────
echo "[4/6] Setting up .env file..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "  ╔══════════════════════════════════════════════════╗"
  echo "  ║  .env created from .env.example                  ║"
  echo "  ║  IMPORTANT: edit it before starting the bot!     ║"
  echo "  ║                                                   ║"
  echo "  ║  nano $APP_DIR/.env                  ║"
  echo "  ╚══════════════════════════════════════════════════╝"
  echo ""
  echo "  Required variables:"
  echo "    DRY_RUN=true              (keep true for testing)"
  echo "    EXCHANGE_API_KEY=...      (Binance testnet key)"
  echo "    EXCHANGE_SECRET_KEY=...   (Binance testnet secret)"
  echo "    EXCHANGE_SANDBOX=true     (use testnet, not real money)"
  echo ""
  read -p "  Press ENTER after editing .env to continue..." _
else
  echo "      .env already exists — skipping."
fi

# ── 5. Create data directories ────────────────────────────
echo "[5/6] Creating data directories..."
mkdir -p data/state data/memory models logs

# ── 6. Build and start ────────────────────────────────────
echo "[6/6] Building and starting services..."
docker compose pull prometheus grafana 2>/dev/null || true
docker compose build --no-cache bot
docker compose up -d

echo ""
echo "======================================================"
echo " Deployment complete!"
echo "======================================================"
echo ""
echo " Dashboard : http://$(hostname -I | awk '{print $1}'):8501"
echo " Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
echo " Grafana   : http://$(hostname -I | awk '{print $1}'):3000"
echo ""
echo " Useful commands:"
echo "   docker compose logs -f bot          # live bot logs"
echo "   docker compose ps                   # service status"
echo "   docker compose down                 # stop everything"
echo "   docker compose restart bot          # restart bot only"
echo ""
echo " Bot state: $APP_DIR/data/state/bot-state.json"
