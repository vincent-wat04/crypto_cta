"""
订单执行器：负责向 Polymarket 发送订单。
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from core.types import Order, OrderStatus, Side
from core.config import PMConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """执行结果。"""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    error: Optional[str] = None


class OrderExecutor:
    """
    订单执行器：负责与 Polymarket CLOB 交互。
    
    支持：
    - Dry-run 模式（模拟执行）
    - 真实下单（需要 py-clob-client）
    """
    
    def __init__(self, config: PMConfig, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self._client = None
        
        if not dry_run:
            self._init_client()
    
    def _init_client(self):
        """初始化 CLOB 客户端。"""
        try:
            import os
            from py_clob_client.client import ClobClient
            
            private_key = os.environ.get(self.config.private_key_env)
            if not private_key:
                raise ValueError(f"Environment variable {self.config.private_key_env} not set")
            
            self._client = ClobClient(
                host=self.config.clob_api_url,
                key=private_key,
            )
            logger.info("CLOB client initialized")
        except ImportError:
            logger.warning("py-clob-client not installed, falling back to dry-run")
            self.dry_run = True
        except Exception as e:
            logger.error(f"Failed to init CLOB client: {e}")
            self.dry_run = True
    
    def place_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        """
        下单。
        
        Args:
            token_id: Polymarket token ID
            side: 买/卖
            price: 价格
            quantity: 数量
            
        Returns:
            ExecutionResult
        """
        if self.dry_run:
            return self._simulate_order(token_id, side, price, quantity)
        
        return self._real_order(token_id, side, price, quantity)
    
    def _simulate_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        """模拟执行（dry-run）。"""
        import uuid
        
        order_id = f"sim_{uuid.uuid4().hex[:8]}"
        logger.info(f"[DRY-RUN] Order: {side.value} {quantity} @ {price} (token={token_id})")
        
        return ExecutionResult(
            success=True,
            order_id=order_id,
            filled_price=price,
            filled_quantity=quantity,
        )
    
    def _real_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        quantity: float,
    ) -> ExecutionResult:
        """真实下单。"""
        if self._client is None:
            return ExecutionResult(
                success=False,
                error="CLOB client not initialized",
            )
        
        try:
            # 创建订单
            order = self._client.create_order(
                token_id=token_id,
                price=price,
                size=quantity,
                side="BUY" if side == Side.BUY else "SELL",
            )
            
            # 提交订单
            result = self._client.post_order(order)
            
            logger.info(f"Order placed: {result}")
            
            return ExecutionResult(
                success=True,
                order_id=result.get("orderID"),
                filled_price=price,
                filled_quantity=quantity,
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单。"""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Cancel order: {order_id}")
            return True
        
        if self._client is None:
            return False
        
        try:
            self._client.cancel(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
    
    def get_open_orders(self) -> list:
        """获取未成交订单。"""
        if self.dry_run:
            return []
        
        if self._client is None:
            return []
        
        try:
            return self._client.get_orders()
        except Exception as e:
            logger.error(f"Get orders failed: {e}")
            return []
