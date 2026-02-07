#!/bin/bash
# 创建干净的虚拟环境，解决依赖冲突

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Setting up clean Python environment"
echo "=========================================="

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 删除旧的 venv（如果存在）
if [ -d ".venv" ]; then
    echo "Removing old .venv..."
    rm -rf .venv
fi

# 创建新的 venv
echo "Creating new virtual environment..."
python3 -m venv .venv

# 激活
source .venv/bin/activate

# 升级 pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装核心依赖（兼容版本）
echo "Installing dependencies..."
pip install \
    'numpy>=1.21,<1.25' \
    'pandas>=1.4,<2.0' \
    'ccxt>=4.0' \
    'pyyaml>=6.0' \
    'websockets>=12.0'

# 可选：安装 PM 客户端
echo ""
echo "Optional: Install Polymarket client for live trading:"
echo "  pip install py-clob-client"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the bot:"
echo "  python scripts/run_bot.py --dry-run"
echo ""
echo "To run backtest:"
echo "  python scripts/run_microstructure_backtest.py --days 1"
