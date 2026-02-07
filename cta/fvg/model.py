"""
FVG 预测模型：聚类 + 监督学习。

流程：
1. 聚类：将 FVG 分为不同类别（大缺口/小缺口、有量/无量、趋势延续/反转）
2. 监督学习：预测 FVG 是否会被填充、填充速度、最优入场方向
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .features import get_feature_columns

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed")


class FVGModel:
    """
    FVG 聚类 + 分类模型。
    
    两阶段：
    1. 无监督聚类 → 识别 FVG 类别
    2. 有监督分类 → 预测填充概率
    """
    
    def __init__(self, n_clusters: int = 4):
        if not HAS_SKLEARN:
            raise ImportError("pip install scikit-learn")
        
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._is_fitted = False
        self._feature_cols = get_feature_columns()
    
    def fit(
        self,
        feature_df: pd.DataFrame,
        target_col: str = "label_filled",
    ) -> Dict[str, Any]:
        """
        训练模型。
        
        Args:
            feature_df: 特征矩阵（由 build_fvg_feature_matrix 生成）
            target_col: 目标列名
        
        Returns:
            训练报告
        """
        X, y = self._prepare_data(feature_df, target_col)
        
        if len(X) < 20:
            logger.warning(f"Too few samples: {len(X)}")
            return {"error": "insufficient_data", "n_samples": len(X)}
        
        # 阶段1：聚类
        logger.info(f"[Phase 1] Clustering {len(X)} FVGs into {self.n_clusters} types...")
        X_scaled = self.scaler.fit_transform(X)
        clusters = self.clusterer.fit_predict(X_scaled)
        
        cluster_report = self._analyze_clusters(feature_df, clusters, y)
        
        # 将聚类标签作为额外特征
        X_with_cluster = np.column_stack([X_scaled, clusters])
        
        # 阶段2：分类
        logger.info(f"[Phase 2] Training classifier (target: {target_col})...")
        
        # 时序交叉验证
        n_splits = min(5, len(X) // 10)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(self.classifier, X_with_cluster, y, cv=tscv, scoring="accuracy")
            cv_accuracy = scores.mean()
            logger.info(f"  CV Accuracy: {cv_accuracy:.4f} (+/- {scores.std():.4f})")
        else:
            cv_accuracy = 0
        
        # 全量训练
        self.classifier.fit(X_with_cluster, y)
        train_pred = self.classifier.predict(X_with_cluster)
        train_accuracy = accuracy_score(y, train_pred)
        
        self._is_fitted = True
        
        # 特征重要性
        feature_names = self._feature_cols + ["cluster"]
        importances = self.classifier.feature_importances_
        top_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        
        report = {
            "n_samples": len(X),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "positive_rate": round(y.mean() * 100, 2),
            "train_accuracy": round(train_accuracy * 100, 2),
            "cv_accuracy": round(cv_accuracy * 100, 2),
            "top_features": top_features,
            "cluster_report": cluster_report,
            "classification_report": classification_report(y, train_pred, output_dict=True),
        }
        
        return report
    
    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        预测 FVG 填充概率。
        
        Returns:
            带有 pred_fill、pred_proba、cluster 列的 DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        X, _ = self._prepare_data(feature_df, target_col=None)
        X_scaled = self.scaler.transform(X)
        clusters = self.clusterer.predict(X_scaled)
        X_with_cluster = np.column_stack([X_scaled, clusters])
        
        pred = self.classifier.predict(X_with_cluster)
        proba = self.classifier.predict_proba(X_with_cluster)
        
        result = feature_df.copy()
        result["pred_fill"] = pred
        result["pred_proba"] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        result["cluster"] = clusters
        
        return result
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """准备特征和标签。"""
        available_cols = [c for c in self._feature_cols if c in df.columns]
        X = df[available_cols].fillna(0).values
        
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].values.astype(int)
        
        # 更新实际使用的特征列
        self._feature_cols = available_cols
        
        return X, y
    
    def _analyze_clusters(
        self,
        df: pd.DataFrame,
        clusters: np.ndarray,
        y: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """分析聚类结果。"""
        report = []
        
        for c in range(self.n_clusters):
            mask = clusters == c
            n = mask.sum()
            if n == 0:
                continue
            
            cluster_y = y[mask]
            fill_rate = cluster_y.mean() * 100
            
            # 该聚类的平均特征
            cluster_df = df.iloc[mask.nonzero()[0]]
            
            info = {
                "cluster": c,
                "n_samples": int(n),
                "fill_rate": round(fill_rate, 2),
                "avg_gap_pct": round(cluster_df["gap_size_pct"].mean(), 4) if "gap_size_pct" in cluster_df else 0,
                "avg_volume_ratio": round(cluster_df["volume_ratio"].mean(), 2) if "volume_ratio" in cluster_df else 0,
                "bullish_pct": round((cluster_df["fvg_type"] == 1).mean() * 100, 1) if "fvg_type" in cluster_df else 0,
            }
            report.append(info)
        
        return report
    
    def save(self, path: Path) -> None:
        """保存模型。"""
        import pickle
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "clusterer": self.clusterer,
                "classifier": self.classifier,
                "feature_cols": self._feature_cols,
                "n_clusters": self.n_clusters,
            }, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FVGModel":
        """加载模型。"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        model = cls(n_clusters=data["n_clusters"])
        model.scaler = data["scaler"]
        model.clusterer = data["clusterer"]
        model.classifier = data["classifier"]
        model._feature_cols = data["feature_cols"]
        model._is_fitted = True
        return model
