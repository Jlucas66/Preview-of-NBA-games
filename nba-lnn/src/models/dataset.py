"""Carrega o dataset processado, separa por temporada e normaliza.

A normalização (média/desvio) e o preenchimento de valores ausentes usam
SOMENTE as temporadas de treino, para não vazar informação de val/teste.
"""
import json
from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class Split:
    seq_home: np.ndarray
    ts_home: np.ndarray
    seq_away: np.ndarray
    ts_away: np.ndarray
    static: np.ndarray
    y: np.ndarray
    elo_prob: np.ndarray
    game_id: np.ndarray

    def __len__(self):
        return len(self.y)


def load_splits():
    path = config.DATA_PROCESSED / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError("dataset.npz não encontrado. Rode: python -m src.features.build")
    d = np.load(path, allow_pickle=False)
    meta = json.loads((config.DATA_PROCESSED / "meta.json").read_text())

    season = d["season"]
    masks = {
        "train": np.isin(season, config.train_seasons()),
        "val": season == config.VAL_SEASON,
        "test": season == config.TEST_SEASON,
    }

    seq_h, seq_a = d["seq_home"].copy(), d["seq_away"].copy()
    static = d["static"].copy()
    tr = masks["train"]

    # valores ausentes nas estáticas -> mediana do treino
    med = np.nanmedian(static[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    idx = np.where(np.isnan(static))
    static[idx] = med[idx[1]]

    # padronização
    seq_train = np.concatenate([seq_h[tr], seq_a[tr]]).reshape(-1, seq_h.shape[-1])
    s_mu, s_sd = seq_train.mean(0), seq_train.std(0) + 1e-6
    st_mu, st_sd = static[tr].mean(0), static[tr].std(0) + 1e-6
    seq_h = (seq_h - s_mu) / s_sd
    seq_a = (seq_a - s_mu) / s_sd
    static = (static - st_mu) / st_sd

    # guardado para o app poder normalizar features editadas do mesmo jeito
    meta["norm"] = {"static_median": med, "static_mu": st_mu, "static_sd": st_sd,
                    "seq_mu": s_mu, "seq_sd": s_sd}

    splits = {}
    for name, m in masks.items():
        splits[name] = Split(
            seq_home=seq_h[m].astype(np.float32), ts_home=d["ts_home"][m],
            seq_away=seq_a[m].astype(np.float32), ts_away=d["ts_away"][m],
            static=static[m].astype(np.float32), y=d["y"][m],
            elo_prob=d["elo_prob"][m], game_id=d["game_id"][m],
        )
    return splits, meta
