"""Métricas de avaliação para previsão probabilística binária."""
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def evaluate(y_true: np.ndarray, p: np.ndarray) -> dict:
    """
    accuracy : % de acertos com limiar 0.5
    log_loss : penaliza probabilidades confiantes e erradas (menor = melhor)
    brier    : erro quadrático médio da probabilidade (menor = melhor)
    auc      : capacidade de ordenar jogos por chance de vitória (maior = melhor)
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y_true, p >= 0.5),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
        "auc": roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan"),
    }
