"""GPU-capable pairwise neural reranker for candidate-level numeric features."""

from __future__ import annotations

from collections import defaultdict
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from alphasql.algorithm.selection.schema_signal import Standardizer


class SchemaAwareMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=(128, 64), dropout: float = 0.15):
        super().__init__()
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def build_pair_indices(records: Sequence[Mapping[str, Any]]):
    """Create positive-negative pairs within each question, emphasizing hard negatives."""
    grouped = defaultdict(lambda: {0: [], 1: []})
    for index, record in enumerate(records):
        grouped[str(record["question_id"])][int(record["label"])].append(index)
    positive_indices, negative_indices, weights = [], [], []
    for candidates in grouped.values():
        for positive in candidates[1]:
            for negative in candidates[0]:
                positive_features = records[positive].get("features", {})
                features = records[negative].get("features", {})
                hardness = 1.0
                hardness += float(features.get("exec_success", 0.0))
                hardness += float(features.get("schema_table_coverage", 0.0))
                hardness += float(features.get("schema_column_coverage", 0.0))
                hardness += float(features.get("exec_result_group_share", 0.0))
                # The reranker exists to repair self-consistency failures.  Give
                # extra weight to a wrong candidate whenever its execution-result
                # cluster is at least as popular as the correct candidate's.
                if float(features.get("exec_result_group_size", 0.0)) >= float(
                    positive_features.get("exec_result_group_size", 0.0)
                ):
                    hardness += 4.0
                if float(features.get("exec_result_group_share", 0.0)) >= float(
                    positive_features.get("exec_result_group_share", 0.0)
                ):
                    hardness += 2.0
                positive_indices.append(positive)
                negative_indices.append(negative)
                weights.append(hardness)
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


class NeuralPairwiseReranker:
    """MLP trained with RankNet plus a small class-balanced pointwise objective."""

    def __init__(
        self,
        hidden_sizes=(128, 64),
        dropout: float = 0.15,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        pointwise_weight: float = 0.25,
        seed: int = 42,
        device: str = "auto",
        patience: int = 40,
    ):
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.pointwise_weight = pointwise_weight
        self.seed = seed
        self.device = self.resolve_device(device)
        self.patience = patience

    @staticmethod
    def resolve_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device(device)

    def fit(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        records: Sequence[Mapping[str, Any]],
        validation_values: np.ndarray | None = None,
        validation_records: Sequence[Mapping[str, Any]] | None = None,
    ):
        values = np.asarray(values, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if len(values) != len(records):
            raise ValueError("Feature rows and records must have the same length")
        if len(set(labels.tolist())) < 2:
            raise ValueError("Training candidates must include both correct and incorrect SQL labels")
        positive, negative, pair_weights = build_pair_indices(records)
        if not len(positive):
            raise ValueError("Pairwise training needs at least one question with both positive and negative candidates")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.scaler = Standardizer().fit(values)
        x = torch.as_tensor(self.scaler.transform(values), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        positive = torch.as_tensor(positive, dtype=torch.long, device=self.device)
        negative = torch.as_tensor(negative, dtype=torch.long, device=self.device)
        pair_weights = torch.as_tensor(pair_weights, dtype=torch.float32, device=self.device)
        validation_tensors = None
        if validation_values is not None and validation_records:
            validation_values = np.asarray(validation_values, dtype=np.float32)
            validation_labels = np.asarray([record["label"] for record in validation_records], dtype=np.float32)
            val_positive, val_negative, val_weights = build_pair_indices(validation_records)
            validation_tensors = (
                torch.as_tensor(self.scaler.transform(validation_values), dtype=torch.float32, device=self.device),
                torch.as_tensor(validation_labels, dtype=torch.float32, device=self.device),
                torch.as_tensor(val_positive, dtype=torch.long, device=self.device),
                torch.as_tensor(val_negative, dtype=torch.long, device=self.device),
                torch.as_tensor(val_weights, dtype=torch.float32, device=self.device),
            )

        self.model = SchemaAwareMLP(values.shape[1], self.hidden_sizes, self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        positive_weight = torch.tensor(
            [(len(labels) - labels.sum()) / max(labels.sum(), 1.0)],
            dtype=torch.float32,
            device=self.device,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.training_history = []
        best_validation_loss = float("inf")
        best_state = None
        stale_epochs = 0
        self.best_epoch = self.epochs
        for epoch in range(self.epochs):
            self.model.train()
            permutation = torch.randperm(len(positive), generator=generator).to(self.device)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(permutation), self.batch_size):
                batch = permutation[start:start + self.batch_size]
                scores = self.model(x)
                rank_losses = F.softplus(-(scores[positive[batch]] - scores[negative[batch]]))
                rank_loss = (rank_losses * pair_weights[batch]).sum() / pair_weights[batch].sum()
                point_loss = F.binary_cross_entropy_with_logits(scores, y, pos_weight=positive_weight)
                loss = rank_loss + self.pointwise_weight * point_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            validation_loss = None
            if validation_tensors is not None:
                self.model.eval()
                val_x, val_y, val_positive, val_negative, val_weights = validation_tensors
                with torch.inference_mode():
                    val_scores = self.model(val_x)
                    if len(val_positive):
                        val_rank_losses = F.softplus(
                            -(val_scores[val_positive] - val_scores[val_negative])
                        )
                        val_rank_loss = (val_rank_losses * val_weights).sum() / val_weights.sum()
                    else:
                        val_rank_loss = torch.tensor(0.0, device=self.device)
                    val_point_loss = F.binary_cross_entropy_with_logits(
                        val_scores, val_y, pos_weight=positive_weight
                    )
                    validation_loss = float(
                        (val_rank_loss + self.pointwise_weight * val_point_loss).cpu()
                    )
                if validation_loss < best_validation_loss - 1e-5:
                    best_validation_loss = validation_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == self.epochs:
                entry = {"epoch": epoch + 1, "loss": epoch_loss / max(1, batches)}
                if validation_loss is not None:
                    entry["validation_loss"] = validation_loss
                self.training_history.append(entry)
            if validation_tensors is not None and stale_epochs >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.pair_count = len(positive)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(
            self.scaler.transform(np.asarray(values, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            return torch.sigmoid(self.model(x)).cpu().numpy()

    def save(self, path: str | Path, feature_names: Sequence[str]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_size": len(feature_names),
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "feature_names": list(feature_names),
            "mean": self.scaler.mean,
            "scale": self.scaler.scale,
            "training_history": self.training_history,
            "best_epoch": self.best_epoch,
        }, path)
