"""
GBM 模型：低复杂度 + early stopping + 概率校准。

使用 HistGradientBoostingClassifier（比 GradientBoostingClassifier 快 10x+）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss, roc_auc_score,
    brier_score_loss, f1_score,
)

logger = logging.getLogger(__name__)


class TrendGBM:
    """
    趋势预测 GBM 模型。
    
    特点：
    - HistGradientBoostingClassifier（原生支持 early stopping）
    - 概率校准（isotonic regression）
    - 实时训练日志
    """

    def __init__(
        self,
        max_iter: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        max_features: float = 0.7,
        l2_regularization: float = 1.0,
        early_stopping: bool = True,
        n_iter_no_change: int = 20,
        validation_fraction: float = 0.15,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        self.params = {
            "max_iter": max_iter,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "l2_regularization": l2_regularization,
            "early_stopping": early_stopping,
            "n_iter_no_change": n_iter_no_change,
            "validation_fraction": validation_fraction,
            "random_state": random_state,
        }
        self.calibrate = calibrate
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.valid_cols: List[int] = []
        self.training_log: Dict = {}

    def _filter_features(self, X: np.ndarray, feature_names: List[str]):
        """移除常数列和高 NaN 列。"""
        valid = []
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.std(col) > 1e-10 and np.isfinite(col).mean() > 0.5:
                valid.append(i)
        self.valid_cols = valid
        self.feature_names = [feature_names[i] for i in valid]
        return X[:, valid]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        训练模型。
        
        Returns:
            training_log dict
        """
        logger.info(f"  [GBM] Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"  [GBM] Params: depth={self.params['max_depth']}, "
                     f"lr={self.params['learning_rate']}, "
                     f"max_iter={self.params['max_iter']}")

        # Filter features
        X_train = self._filter_features(X_train.copy(), feature_names)
        X_val = X_val[:, self.valid_cols]
        logger.info(f"  [GBM] Valid features: {len(self.feature_names)}/{len(feature_names)}")

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # Train with class weights
        n_classes = len(np.unique(y_train))
        if n_classes > 2:
            # 为多分类计算 sample_weight
            class_counts = np.bincount(y_train.astype(int))
            total = len(y_train)
            weights = np.array([total / (n_classes * c) if c > 0 else 1.0 for c in class_counts])
            sample_w = weights[y_train.astype(int)]
            logger.info(f"  [GBM] Class weights: {dict(enumerate(np.round(weights, 3)))}")
        else:
            sample_w = None

        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(X_train_s, y_train, sample_weight=sample_w)

        n_iter = self.model.n_iter_
        logger.info(f"  [GBM] Converged at {n_iter} iterations")

        # Evaluate
        train_pred = self.model.predict(X_train_s)
        val_pred = self.model.predict(X_val_s)
        val_proba = self.model.predict_proba(X_val_s)

        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()

        logger.info(f"  [GBM] Train acc: {train_acc:.4f}")
        logger.info(f"  [GBM] Val acc:   {val_acc:.4f}")
        logger.info(f"  [GBM] Overfit:   {train_acc - val_acc:.4f}")

        # 在 calibration 之前保存原始模型引用（用于 feature_importances_）
        raw_model = self.model

        # Calibrate
        if self.calibrate and len(X_val_s) > 10:
            try:
                cal = CalibratedClassifierCV(self.model, cv="prefit", method="isotonic")
                cal.fit(X_val_s, y_val)
                self.model = cal
                logger.info("  [GBM] Calibrated (isotonic)")
            except Exception as e:
                logger.warning(f"  [GBM] Calibration failed: {e}")

        # Metrics
        val_metrics = {"val_accuracy": round(val_acc, 4)}
        try:
            if val_proba.shape[1] == 2:
                val_metrics["val_roc_auc"] = round(
                    roc_auc_score(y_val, val_proba[:, 1]), 4
                )
                val_metrics["val_brier"] = round(
                    brier_score_loss(y_val, val_proba[:, 1]), 4
                )
            val_metrics["val_log_loss"] = round(log_loss(y_val, val_proba), 4)
            val_metrics["val_f1"] = round(
                f1_score(y_val, val_pred, average="weighted"), 4
            )
        except Exception:
            pass

        logger.info(f"  [GBM] Val metrics: {val_metrics}")

        # Feature importance — HistGradientBoostingClassifier 在部分 sklearn
        # 版本中不暴露 feature_importances_，使用 permutation_importance 代替
        self._raw_model = raw_model
        try:
            perm_result = permutation_importance(
                raw_model, X_val_s, y_val,
                n_repeats=5, random_state=42, n_jobs=-1,
                scoring="accuracy",
            )
            importances = perm_result.importances_mean
            # 负值截断为 0
            importances = np.maximum(importances, 0)
        except Exception as e:
            logger.warning(f"  [GBM] Permutation importance failed: {e}")
            importances = np.zeros(len(self.feature_names))

        feat_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )

        regime_imp = sum(
            imp for name, imp in zip(self.feature_names, importances)
            if "regime" in name.lower()
        )
        total_imp = sum(importances) + 1e-10

        logger.info("  [GBM] Top 10 features:")
        for name, imp in feat_imp[:10]:
            logger.info(f"    {name}: {imp:.4f}")
        logger.info(f"  [GBM] Regime importance: {regime_imp/total_imp:.1%}")

        self.training_log = {
            "model_type": "HistGradientBoostingClassifier",
            "params": self.params,
            "n_iter_actual": n_iter,
            "n_features": len(self.feature_names),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "overfit_gap": round(train_acc - val_acc, 4),
            "val_metrics": val_metrics,
            "regime_importance_ratio": round(regime_imp / total_imp, 4),
            "top_features": [(n, round(float(v), 6)) for n, v in feat_imp[:20]],
        }

        return self.training_log

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测：返回 (predictions, probabilities)。"""
        X_valid = X[:, self.valid_cols]
        X_s = self.scaler.transform(X_valid)
        preds = self.model.predict(X_s)
        proba = self.model.predict_proba(X_s)
        return preds, proba

    def save_log(self, log_dir: Path, prefix: str = "gbm"):
        """保存训练日志。"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"{prefix}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.training_log, f, indent=2, default=str)
        logger.info(f"  [GBM] Log saved: {path}")
        return path
