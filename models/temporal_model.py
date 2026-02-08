"""
时序模型：LSTM/GRU 用于趋势预测。

输入：(batch, lookback, n_features) 的 3D 序列
输出：(batch, n_classes) 分类概率

特点：
- 支持 LSTM 和 GRU
- Dropout + BatchNorm 防过拟合
- 实时训练日志（每 epoch 输出到 terminal）
- 支持 early stopping
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, temporal model disabled")


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int = 60,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 2D 特征矩阵转为 3D 序列。
    
    Args:
        features: (n_bars, n_features)
        labels: (n_bars,)
        lookback: 回溯窗口
        step: 滑动步长

    Returns:
        X: (n_samples, lookback, n_features)
        y: (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(features), step):
        X.append(features[i - lookback:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


if HAS_TORCH:

    class LSTMClassifier(nn.Module):
        """LSTM 分类器。"""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            n_classes: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = False,
            cell_type: str = "lstm",
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            RNNClass = nn.LSTM if cell_type == "lstm" else nn.GRU
            self.rnn = RNNClass(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )

            fc_input = hidden_size * (2 if bidirectional else 1)
            self.bn = nn.BatchNorm1d(fc_input)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(fc_input, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.rnn(x)
            # 取最后一个时步
            last = out[:, -1, :]
            last = self.bn(last)
            last = self.dropout(last)
            return self.fc(last)


class TrendLSTM:
    """
    LSTM 趋势预测模型（封装训练 + 推理）。
    """

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        cell_type: str = "lstm",
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
    ):
        if not HAS_TORCH:
            raise ImportError("pip install torch")

        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.lr = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.n_classes = 2
        self.training_log: Dict = {}
        self.epoch_logs: List[Dict] = []

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score normalization on feature dim."""
        if fit:
            # 按特征列算 mean/std（collapse time + sample dims）
            self.scaler_mean = np.nanmean(X.reshape(-1, X.shape[-1]), axis=0)
            self.scaler_std = np.nanstd(X.reshape(-1, X.shape[-1]), axis=0) + 1e-8
        X_norm = (X - self.scaler_mean) / self.scaler_std
        return np.nan_to_num(X_norm, 0)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None,  # noqa: ARG002 — reserved for future use
    ) -> Dict:
        """
        训练 LSTM。实时在 terminal 输出 epoch 进度。
        
        Args:
            X_train/X_val: (n, lookback, n_features) 3D
            y_train/y_val: (n,) labels
        """
        self.n_classes = len(np.unique(y_train))
        n_features = X_train.shape[-1]

        logger.info(f"  [LSTM] Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"  [LSTM] Lookback={self.lookback}, Hidden={self.hidden_size}, "
                     f"Layers={self.num_layers}, Cell={self.cell_type}")
        logger.info(f"  [LSTM] Features: {n_features}, Classes: {self.n_classes}")

        # Normalize
        X_train = self._normalize(X_train, fit=True)
        X_val = self._normalize(X_val)

        # To tensors
        device = torch.device("cpu")
        X_tr = torch.FloatTensor(X_train).to(device)
        y_tr = torch.LongTensor(y_train).to(device)
        X_v = torch.FloatTensor(X_val).to(device)
        y_v = torch.LongTensor(y_val).to(device)

        train_ds = TensorDataset(X_tr, y_tr)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Model
        self.model = LSTMClassifier(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            n_classes=self.n_classes,
            dropout=self.dropout,
            cell_type=self.cell_type,
        ).to(device)

        # Class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=self.n_classes).astype(float)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        weight_tensor = torch.FloatTensor(class_weights).to(device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7,
        )

        # Training loop with live logging
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.epoch_logs = []

        logger.info(f"  [LSTM] Training (max {self.max_epochs} epochs, "
                     f"patience={self.patience})...")
        logger.info("  ┌─────────┬───────────┬───────────┬─────────┬─────────┬────────┐")
        logger.info("  │  Epoch  │ Train Loss│  Val Loss │ Tr Acc  │ Val Acc │   LR   │")
        logger.info("  ├─────────┼───────────┼───────────┼─────────┼─────────┼────────┤")

        t0 = time.time()

        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(xb)
                train_correct += (out.argmax(dim=1) == yb).sum().item()
                train_total += len(xb)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_v)
                val_loss = criterion(val_out, y_v).item()
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred == y_v).float().mean().item()

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            # Live log
            logger.info(
                f"  │  {epoch+1:>4}   │  {train_loss:>7.4f}  │  {val_loss:>7.4f}  │"
                f" {train_acc:>5.1%}  │ {val_acc:>5.1%}  │{current_lr:>7.5f}│"
            )

            self.epoch_logs.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "lr": current_lr,
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"  └─── Early stopped at epoch {epoch+1} ─────────────────────┘")
                    break
        else:
            logger.info(f"  └─── Completed {self.max_epochs} epochs ──────────────────────┘")

        elapsed = time.time() - t0
        logger.info(f"  [LSTM] Training took {elapsed:.1f}s")

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            val_out = self.model(X_v)
            val_pred = val_out.argmax(dim=1).cpu().numpy()
            final_val_acc = (val_pred == y_val).mean()

        self.training_log = {
            "model_type": f"{self.cell_type.upper()}_Classifier",
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "n_features": n_features,
            "n_classes": self.n_classes,
            "best_epoch": len(self.epoch_logs) - patience_counter,
            "total_epochs": len(self.epoch_logs),
            "best_val_loss": round(best_val_loss, 5),
            "final_val_acc": round(final_val_acc, 4),
            "training_time_s": round(elapsed, 1),
            "epoch_history": self.epoch_logs,
        }

        logger.info(f"  [LSTM] Best val loss: {best_val_loss:.4f} at epoch "
                     f"{self.training_log['best_epoch']}")
        logger.info(f"  [LSTM] Final val acc: {final_val_acc:.4f}")

        return self.training_log

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """推理：返回 (predictions, probabilities)。"""
        X_norm = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_norm)
            out = self.model(X_t)
            proba = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
        return preds, proba

    def save_log(self, log_dir: Path, prefix: str = "lstm"):
        """保存训练日志。"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"{prefix}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.training_log, f, indent=2, default=str)
        logger.info(f"  [LSTM] Log saved: {path}")
        return path
