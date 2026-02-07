"""
配置管理：统一的配置系统。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import os

import yaml


@dataclass
class ExchangeConfig:
    """交易所配置。"""
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "SOL/USDT"
    # WebSocket 设置
    ws_url: str = "wss://stream.binance.com:9443/ws"
    # 数据设置
    trade_buffer_size: int = 10000  # 逐笔成交缓冲区大小
    orderbook_depth: int = 20


@dataclass
class StrategyConfig:
    """策略配置。"""
    # Price Impact
    impact_threshold_pct: float = 0.3
    impact_window_ms: int = 5000
    impact_volume_ratio: float = 2.0
    
    # Spread
    spread_percentile_threshold: float = 85.0
    spread_lookback: int = 500
    
    # Order Flow
    imbalance_threshold: float = 0.6
    imbalance_window: int = 100
    
    # MM Refill
    refill_window_ms: int = 30000
    spread_recovery_threshold: float = 50.0
    price_hold_tolerance_pct: float = 0.15
    
    # Signal
    min_confidence: float = 0.5
    hold_time_ms: int = 60000


@dataclass
class PMConfig:
    """Polymarket 配置。"""
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    private_key_env: str = "POLYMARKET_PRIVATE_KEY"
    # 市场偏好
    preferred_durations: list = field(default_factory=lambda: ["15min", "1h"])
    keywords: list = field(default_factory=lambda: ["solana", "sol up", "sol down"])
    # 交易设置
    entry_price_max: float = 0.15  # 最高入场价（劣势方）
    take_profit_mult: float = 3.0  # 止盈倍数


@dataclass
class BotConfig:
    """Bot 配置。"""
    dry_run: bool = True
    poll_interval_ms: int = 1000  # 主循环间隔
    max_position_size: float = 100.0  # 最大持仓
    # 风控
    max_daily_loss: float = 50.0
    max_drawdown_pct: float = 10.0
    # 性能
    log_level: str = "INFO"
    enable_profiling: bool = False


@dataclass
class Config:
    """总配置。"""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    pm: PMConfig = field(default_factory=PMConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            exchange=ExchangeConfig(**d.get("exchange", {})),
            strategy=StrategyConfig(**d.get("strategy", {})),
            pm=PMConfig(**d.get("pm", {})),
            bot=BotConfig(**d.get("bot", {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exchange": self.exchange.__dict__,
            "strategy": self.strategy.__dict__,
            "pm": self.pm.__dict__,
            "bot": self.bot.__dict__,
        }


def load_config(path: Optional[Path] = None) -> Config:
    """加载配置文件。"""
    if path is None:
        # 默认路径
        path = Path(__file__).resolve().parents[1] / "config.yaml"
    
    if not path.exists():
        print(f"Config file not found: {path}, using defaults")
        return Config()
    
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    
    return Config.from_dict(data)


def save_config(config: Config, path: Path) -> None:
    """保存配置到文件。"""
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
