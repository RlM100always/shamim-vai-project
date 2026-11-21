from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


@dataclass
class TorchPredictor:
    model: object
    device: object

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            positive = torch.sigmoid(self.model(tensor).reshape(-1)).cpu().numpy()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_logistic_regression(C: float):
    return LogisticRegression(
        C=float(C), max_iter=2000, solver="liblinear",
        class_weight="balanced", random_state=42,
    )


def make_linear_svm(C: float):
    return LinearSVC(
        C=float(C), class_weight="balanced",
        max_iter=5000, random_state=42,
    )


def _build_torch_model(kind: str, stage: str, input_dim: int):
    import torch.nn as nn

    refined = stage == "refined"
    if kind == "dnn":
        hidden = [256, 128] if refined else [128]
        dropout = 0.35 if refined else 0.20
        layers = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    if kind == "residual_mlp":
        width = 256 if refined else 128
        dropout = 0.25 if refined else 0.15
        blocks = 2 if refined else 1

        class ResidualBlock(nn.Module):
            def __init__(self, size: int, drop: float):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(size, size),
                    nn.LayerNorm(size),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(size, size),
                    nn.LayerNorm(size),
                )

            def forward(self, x):
                return torch.relu(x + self.layers(x))

        class ResidualMLP(nn.Module):
            def __init__(self, n_features: int):
                super().__init__()
                self.projection = nn.Sequential(
                    nn.Linear(n_features, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.blocks = nn.Sequential(
                    *[ResidualBlock(width, dropout) for _ in range(blocks)]
                )
                self.output = nn.Linear(width, 1)

            def forward(self, x):
                return self.output(self.blocks(self.projection(x)))

        return ResidualMLP(input_dim)

    raise ValueError(f"Unknown deep model: {kind}")


def train_deep_model(
    kind: str,
    stage: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    seed: int = 42,
):
    """Train a baseline or refined feed-forward network with early stopping."""
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_torch_model(kind, stage, X_train.shape[1]).to(device)

    refined = stage == "refined"
    epochs = 24 if refined else 10
    patience = 5 if refined else 3
    learning_rate = 5e-4 if refined else 1e-3
    batch_size = 64
    weight_decay = 1e-4 if refined else 0.0

    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    Xv = torch.as_tensor(X_valid, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_valid, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).reshape(-1), yb)
            loss.backward()
            if refined:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv).reshape(-1), yv).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    predictor = TorchPredictor(model=model, device=device)
    validation_predictions = predictor.predict(X_valid)
    validation_f1 = float(f1_score(y_valid, validation_predictions, zero_division=0))
    settings = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": epochs,
        "best_epoch": best_epoch,
        "patience": patience,
        "weight_decay": weight_decay,
        "hidden_units": "256-128" if kind == "dnn" and refined else (
            "128" if kind == "dnn" else ("256 x 2 residual blocks" if refined else "128 x 1 residual block")
        ),
        "dropout": 0.35 if refined and kind == "dnn" else (
            0.20 if kind == "dnn" else (0.25 if refined else 0.15)
        ),
    }
    return predictor, validation_f1, settings
