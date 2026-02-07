"""
FVG (Fair Value Gap) 策略模块。

- detector: FVG 检测
- features: V1 逐笔成交特征
- features_v3: V3 基于 indicators/ 库的全面特征
- orderbook_features: orderbook 特征
- model: 聚类 + 分类模型
"""
from .detector import detect_fvg, FVG, FVGType, FVGStatus
from .features import build_fvg_feature_matrix, get_feature_columns
from .features_v3 import (
    build_fvg_features_v3,
    add_triple_barrier_labels,
    get_v3_feature_columns,
)
from .orderbook_features import add_orderbook_features, get_ob_feature_columns
from .model import FVGModel

__all__ = [
    "detect_fvg",
    "FVG",
    "FVGType",
    "FVGStatus",
    "build_fvg_feature_matrix",
    "get_feature_columns",
    "build_fvg_features_v3",
    "add_triple_barrier_labels",
    "get_v3_feature_columns",
    "add_orderbook_features",
    "get_ob_feature_columns",
    "FVGModel",
]
