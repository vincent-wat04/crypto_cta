#!/usr/bin/env python3
"""
实时交易 Bot：接收交易所高频数据，生成信号，执行 Polymarket 交易。

运行方式：
    .venv/bin/python scripts/run_bot.py --dry-run
    
实盘运行：
    export POLYMARKET_PRIVATE_KEY=your_private_key
    .venv/bin/python scripts/run_bot.py --live
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core.config import load_config, Config
from core.types import Signal, Side
from engine.realtime import RealtimeEngine
from engine.executor import OrderExecutor
from src.pm.pm_loader import list_solana_markets  # noqa: Polymarket still in src/pm/

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TradingBot")


class TradingBot:
    """
    实时交易 Bot。
    
    整合：
    1. 交易所实时数据（WebSocket）
    2. 微观结构信号生成
    3. Polymarket 市场发现
    4. 订单执行
    """
    
    def __init__(self, config: Config, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        
        # 组件
        self._engine = RealtimeEngine(config, on_signal=self._on_signal)
        self._executor = OrderExecutor(config.pm, dry_run=dry_run)
        
        # 状态
        self._pm_markets = []
        self._last_market_refresh = 0
        self._signal_queue: asyncio.Queue[Signal] = asyncio.Queue()
        
        logger.info(f"TradingBot initialized (dry_run={dry_run})")
    
    async def run(self):
        """运行 Bot。"""
        logger.info("Starting TradingBot...")
        
        # 初始加载 PM 市场
        await self._refresh_pm_markets()
        
        # 启动任务
        tasks = [
            asyncio.create_task(self._engine.run()),
            asyncio.create_task(self._signal_processor()),
            asyncio.create_task(self._market_refresher()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled")
        except KeyboardInterrupt:
            logger.info("Bot interrupted")
        finally:
            self._engine.stop()
    
    def _on_signal(self, signal: Signal):
        """信号回调（同步）。"""
        logger.info(f"🚨 Signal received: {signal.signal_type.name} @ {signal.price:.4f} (conf={signal.confidence:.2f})")
        
        # 放入队列异步处理
        try:
            self._signal_queue.put_nowait(signal)
        except asyncio.QueueFull:
            logger.warning("Signal queue full, dropping signal")
    
    async def _signal_processor(self):
        """异步处理信号。"""
        while True:
            try:
                signal = await asyncio.wait_for(
                    self._signal_queue.get(),
                    timeout=1.0,
                )
                await self._process_signal(signal)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal processing error: {e}")
    
    async def _process_signal(self, signal: Signal):
        """处理单个信号。"""
        # 找到合适的 PM 市场
        market = self._find_pm_market(signal)
        if not market:
            logger.warning("No suitable PM market found")
            return
        
        # 确定交易方向
        # 如果信号是 LONG（预期上涨），买入 "Up" outcome
        # 如果信号是 SHORT（预期下跌），买入 "Down" outcome
        if signal.is_long:
            token_id = market.get("up_token_id")
            outcome = "Up"
        else:
            token_id = market.get("down_token_id")
            outcome = "Down"
        
        if not token_id:
            logger.warning(f"Token ID not found for {outcome}")
            return
        
        # 获取当前价格
        outcome_price = market.get(f"{outcome.lower()}_price", 0.5)
        
        # 检查是否满足入场条件（劣势方）
        if outcome_price > self.config.pm.entry_price_max:
            logger.info(f"{outcome} price {outcome_price:.3f} > max {self.config.pm.entry_price_max:.3f}, skipping")
            return
        
        # 执行订单
        quantity = min(
            self.config.bot.max_position_size / outcome_price,
            100,  # 最大单量
        )
        
        result = self._executor.place_order(
            token_id=token_id,
            side=Side.BUY,
            price=outcome_price,
            quantity=quantity,
        )
        
        if result.success:
            logger.info(f"Order placed: {outcome} @ {outcome_price:.3f}, qty={quantity:.2f}")
        else:
            logger.error(f"Order failed: {result.error}")
    
    def _find_pm_market(self, signal: Signal) -> dict:
        """找到合适的 PM 市场。"""
        if not self._pm_markets:
            return None
        
        # 优先选择最近的市场
        for market in self._pm_markets:
            if market.get("accepting_orders", False):
                return market
        
        return self._pm_markets[0] if self._pm_markets else None
    
    async def _refresh_pm_markets(self):
        """刷新 PM 市场列表。"""
        try:
            markets = list_solana_markets(
                preferred_timeframes=self.config.pm.preferred_durations,
            )
            
            if markets:
                self._pm_markets = markets
                logger.info(f"Loaded {len(markets)} PM markets")
                
                for m in markets[:3]:
                    logger.info(f"  - {m.get('question', 'Unknown')[:50]}")
            else:
                logger.warning("No PM markets found")
                
        except Exception as e:
            logger.error(f"Failed to refresh PM markets: {e}")
    
    async def _market_refresher(self):
        """定期刷新市场。"""
        while True:
            await asyncio.sleep(300)  # 每 5 分钟刷新
            await self._refresh_pm_markets()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="实时交易 Bot")
    parser.add_argument("--dry-run", action="store_true", default=True, help="模拟运行（默认）")
    parser.add_argument("--live", action="store_true", help="实盘运行")
    parser.add_argument("--config", type=str, help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # 确定运行模式
    dry_run = not args.live
    
    if not dry_run:
        logger.warning("=" * 50)
        logger.warning("LIVE MODE - Real orders will be placed!")
        logger.warning("=" * 50)
    
    # 创建并运行 Bot
    bot = TradingBot(config, dry_run=dry_run)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")


if __name__ == "__main__":
    asyncio.run(main())
