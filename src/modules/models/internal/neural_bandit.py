"""
Neural Contextual Bandits for Diabetes Treatment Selection
Copied from ML project — import paths updated for FastAPI integration.

Only change: src.data_generator → src.modules.models.internal.constants
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
logger = logging.getLogger(__name__)
import os
import copy

from src.modules.models.internal.constants import N_TREATMENTS, IDX_TO_TREATMENT


# ─── RewardNetwork ───

class RewardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.1, n_treatments: int = N_TREATMENTS):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.n_treatments = n_treatments
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim), nn.Dropout(dropout)])
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]
        self.heads = nn.ModuleList([nn.Linear(self.feature_dim, 1) for _ in range(n_treatments)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        outputs = [head(features).squeeze(-1) for head in self.heads]
        return torch.stack(outputs, dim=1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)


# ─── NeuralBanditBase ───

class NeuralBanditBase:
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256, device: str = "auto"):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.network = RewardNetwork(input_dim, self.hidden_dims, dropout).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        logger.info(f"{self.__class__.__name__} initialized: input={input_dim}, hidden={self.hidden_dims}, device={self.device}")

    def train(self, X: np.ndarray, actions: np.ndarray, rewards: np.ndarray, epochs: int = 50, val_fraction: float = 0.1, early_stopping_patience: int = 10, verbose: bool = True) -> Dict:
        n = X.shape[0]
        n_val = int(n * val_fraction)
        indices = np.random.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_t = torch.FloatTensor(X[train_idx]).to(self.device)
        a_t = torch.LongTensor(actions[train_idx]).to(self.device)
        r_t = torch.FloatTensor(rewards[train_idx]).to(self.device)
        X_v = torch.FloatTensor(X[val_idx]).to(self.device)
        a_v = torch.LongTensor(actions[val_idx]).to(self.device)
        r_v = torch.FloatTensor(rewards[val_idx]).to(self.device)
        train_ds = TensorDataset(X_t, a_t, r_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, a_batch, r_batch in train_loader:
                self.optimizer.zero_grad()
                pred_all = self.network(X_batch)
                pred = pred_all[torch.arange(len(a_batch)), a_batch]
                loss = nn.MSELoss()(pred, r_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            self._train_losses.append(train_loss)
            self.network.eval()
            with torch.no_grad():
                pred_val = self.network(X_v)
                pred_val_obs = pred_val[torch.arange(len(a_v)), a_v]
                val_loss = nn.MSELoss()(pred_val_obs, r_v).item()
            self._val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.network.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch + 1:>3}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if patience_counter >= early_stopping_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        logger.info(f"Training complete: best_val_loss={best_val_loss:.4f}, epochs_run={epoch + 1}")
        return {"best_val_loss": best_val_loss, "epochs_run": epoch + 1, "train_losses": self._train_losses, "val_losses": self._val_losses}

    def predict_rewards(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            preds = self.network(X_t).cpu().numpy()
        return preds

    def predict_rewards_single(self, x: np.ndarray) -> np.ndarray:
        return self.predict_rewards(x.reshape(1, -1)).flatten()

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        raise NotImplementedError

    def select_actions(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.select_action(X[i])[0] for i in range(X.shape[0])])

    def evaluate(self, X: np.ndarray, counterfactuals: np.ndarray, optimal_actions: Optional[np.ndarray] = None) -> Dict:
        actions = self.select_actions(X)
        n = len(X)
        policy_value = counterfactuals[np.arange(n), actions].mean()
        oracle_value = counterfactuals.max(axis=1).mean()
        result = {"policy_value": round(policy_value, 4), "oracle_value": round(oracle_value, 4), "regret": round(oracle_value - policy_value, 4)}
        if optimal_actions is not None:
            result["accuracy"] = round((actions == optimal_actions).mean(), 4)
        for k in range(N_TREATMENTS):
            result[f"pct_{IDX_TO_TREATMENT[k]}"] = round((actions == k).mean(), 4)
        return result

    def save(self, path: str = "models/neural_bandit.pt") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"network_state": self.network.state_dict(), "optimizer_state": self.optimizer.state_dict(), "config": {"input_dim": self.input_dim, "hidden_dims": self.hidden_dims, "dropout": self.dropout, "lr": self.lr, "weight_decay": self.weight_decay}, "train_losses": self._train_losses, "val_losses": self._val_losses}, path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._train_losses = checkpoint.get("train_losses", [])
        self._val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Loaded model from {path}")


# ─── NeuralThompson ───

class NeuralThompson(NeuralBanditBase):
    def __init__(self, input_dim: int, reg_lambda: float = 1.0, noise_variance: float = 0.25, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.reg_lambda = reg_lambda
        self.noise_variance = noise_variance
        feat_dim = self.network.feature_dim
        self.A = [self.reg_lambda * np.eye(feat_dim) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(feat_dim) / self.reg_lambda for _ in range(N_TREATMENTS)]
        self.b = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]
        self.mu = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]

    def update_posterior(self, x: np.ndarray, action: int, reward: float) -> None:
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()
        k = action
        self.A[k] += np.outer(phi, phi)
        self.b[k] += reward * phi
        phi_col = phi.reshape(-1, 1)
        A_inv = self.A_inv[k]
        numerator = A_inv @ phi_col @ phi_col.T @ A_inv
        denominator = 1.0 + phi_col.T @ A_inv @ phi_col
        self.A_inv[k] = A_inv - numerator / denominator.item()
        self.mu[k] = self.A_inv[k] @ self.b[k]

    def select_action(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()
        sampled_rewards = np.zeros(N_TREATMENTS)
        for k in range(N_TREATMENTS):
            cov = self.noise_variance * self.A_inv[k]
            cov = (cov + cov.T) / 2
            cov += 1e-6 * np.eye(len(cov))
            try:
                theta_k = np.random.multivariate_normal(self.mu[k], cov)
            except np.linalg.LinAlgError:
                theta_k = self.mu[k]
            sampled_rewards[k] = phi @ theta_k
        return int(np.argmax(sampled_rewards)), sampled_rewards

    def compute_confidence(self, x: np.ndarray, n_draws: int = 200) -> Dict:
        self.network.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x.reshape(1, -1)).to(self.device)
            phi = self.network.get_features(x_t).cpu().numpy().flatten()
        covs = []
        for k in range(N_TREATMENTS):
            cov = self.noise_variance * self.A_inv[k]
            cov = (cov + cov.T) / 2
            cov += 1e-6 * np.eye(len(cov))
            covs.append(cov)
        win_counts = np.zeros(N_TREATMENTS)
        for _ in range(n_draws):
            sampled_rewards = np.zeros(N_TREATMENTS)
            for k in range(N_TREATMENTS):
                try:
                    theta_k = np.random.multivariate_normal(self.mu[k], covs[k])
                except np.linalg.LinAlgError:
                    theta_k = self.mu[k]
                sampled_rewards[k] = phi @ theta_k
            winner = int(np.argmax(sampled_rewards))
            win_counts[winner] += 1
        win_rates = win_counts / n_draws
        posterior_means = np.array([self.mu[k] @ phi for k in range(N_TREATMENTS)])
        sorted_means = np.sort(posterior_means)[::-1]
        mean_gap = sorted_means[0] - sorted_means[1]
        recommended_idx = int(np.argmax(win_rates))
        recommended = IDX_TO_TREATMENT[recommended_idx]
        recommended_win_rate = win_rates[recommended_idx]
        confidence_pct = int(round(recommended_win_rate * 100))
        if confidence_pct >= 85:
            confidence_label = "HIGH"
        elif confidence_pct >= 60:
            confidence_label = "MODERATE"
        else:
            confidence_label = "LOW"
        return {
            "win_rates": {IDX_TO_TREATMENT[k]: round(float(win_rates[k]), 3) for k in range(N_TREATMENTS)},
            "recommended": recommended, "recommended_idx": recommended_idx,
            "recommended_win_rate": round(float(recommended_win_rate), 3),
            "confidence_pct": confidence_pct, "confidence_label": confidence_label,
            "posterior_means": {IDX_TO_TREATMENT[k]: round(float(posterior_means[k]), 2) for k in range(N_TREATMENTS)},
            "mean_gap": round(float(mean_gap), 2), "n_draws": n_draws,
        }

    def online_update(self, x: np.ndarray, action: int, reward: float) -> None:
        self.update_posterior(x, action, reward)

    def reset_posterior(self) -> None:
        feat_dim = self.network.feature_dim
        self.A = [self.reg_lambda * np.eye(feat_dim) for _ in range(N_TREATMENTS)]
        self.A_inv = [np.eye(feat_dim) / self.reg_lambda for _ in range(N_TREATMENTS)]
        self.b = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]
        self.mu = [np.zeros(feat_dim) for _ in range(N_TREATMENTS)]

    def save(self, path: str = "models/neural_thompson.pt") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "config": {"input_dim": self.input_dim, "hidden_dims": self.hidden_dims, "dropout": self.dropout, "lr": self.lr, "weight_decay": self.weight_decay, "reg_lambda": self.reg_lambda, "noise_variance": self.noise_variance},
            "train_losses": self._train_losses, "val_losses": self._val_losses,
            "A": [a.copy() for a in self.A], "A_inv": [a.copy() for a in self.A_inv],
            "b": [b.copy() for b in self.b], "mu": [m.copy() for m in self.mu],
        }, path)
        logger.info(f"Saved NeuralThompson (with posterior) to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._train_losses = checkpoint.get("train_losses", [])
        self._val_losses = checkpoint.get("val_losses", [])
        if "A" in checkpoint and "mu" in checkpoint:
            self.A = checkpoint["A"]
            self.A_inv = checkpoint["A_inv"]
            self.b = checkpoint["b"]
            self.mu = checkpoint["mu"]
            logger.info(f"Loaded NeuralThompson (with posterior) from {path}")
        else:
            logger.warning(f"Loaded NeuralThompson from {path} — no posterior found, using fresh prior.")