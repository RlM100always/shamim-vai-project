from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
from sklearn.naive_bayes import MultinomialNB
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


def build_traditional_models() -> dict[str, object]:
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    }


class _ResidualBlock:
    @staticmethod
    def build(width: int, dropout: float):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )


def _build_torch_model(kind: str, input_dim: int):
    import torch.nn as nn

    if kind == "dnn":
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    if kind == "residual_mlp":
        class ResidualMLP(nn.Module):
            def __init__(self, n_features: int):
                super().__init__()
                self.input_layer = nn.Sequential(
                    nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.25)
                )
                self.block1 = _ResidualBlock.build(256, 0.20)
                self.block2 = _ResidualBlock.build(256, 0.20)
                self.activation = nn.ReLU()
                self.output_layer = nn.Linear(256, 1)

            def forward(self, x):
                x = self.input_layer(x)
                x = self.activation(x + self.block1(x))
                x = self.activation(x + self.block2(x))
                return self.output_layer(x)
        return ResidualMLP(input_dim)

    raise ValueError(f"Unknown deep model kind: {kind}")


def train_deep_model(
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
) -> TorchPredictor:
    """Train a PyTorch feed-forward model with validation-based early stopping."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_torch_model(kind, X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_valid = torch.as_tensor(X_valid, dtype=torch.float32, device=device)
    y_valid_tensor = torch.as_tensor(y_valid, dtype=torch.float32, device=device)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb).reshape(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_valid).reshape(-1), y_valid_tensor).item())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return TorchPredictor(model=model, device=device)
