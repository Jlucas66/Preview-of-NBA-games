"""Baselines não neurais para comparar com a rede líquida.

    mandante   - sempre prevê a taxa de vitória do mandante no treino
    elo        - probabilidade dada pelo rating Elo (com vantagem de casa)
    logistica  - regressão logística com as features estáticas + média
                 dos últimos jogos de cada time (mesma informação da rede,
                 mas sem modelar a sequência)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.dataset import Split


def _flat(s: Split) -> np.ndarray:
    return np.concatenate([s.static, s.seq_home.mean(1), s.seq_away.mean(1),
                           s.seq_home[:, -3:].mean(1), s.seq_away[:, -3:].mean(1)], axis=1)


def run_baselines(train: Split, test: Split) -> dict[str, np.ndarray]:
    preds = {}
    preds["mandante"] = np.full(len(test), train.y.mean())
    preds["elo"] = test.elo_prob.astype(float)

    lr = LogisticRegression(C=0.5, max_iter=2000)
    lr.fit(_flat(train), train.y)
    preds["logistica"] = lr.predict_proba(_flat(test))[:, 1]
    return preds
