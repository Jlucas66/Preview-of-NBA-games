"""Figuras e tabelas para o artigo.

Uso (depois de `python -m src.train --model all`):
    python -m src.analysis

Gera em artifacts/figures/:
    comparacao_modelos.png   log-loss e acurácia de cada modelo no teste
    curvas_treino.png        log-loss de validação por época (redes)
    calibracao.png           probabilidade prevista x frequência real (CfC vs Elo)
    importancia.png          importância por permutação das features (CfC)
    efeito_desfalque.png     como a previsão muda com a força de quem joga hoje
e artifacts/importancia.csv.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.models.dataset import load_splits  # noqa: E402
from src.models.metrics import evaluate  # noqa: E402
from src.models.networks import TwoTowerNet  # noqa: E402
from src.train import predict, to_loader  # noqa: E402

FIG = config.ARTIFACTS / "figures"
NET_COLOR = {"cfc": "#2a6fdb", "ncp": "#7b4bc4", "ltc": "#1a9e77", "lstm": "#d95f02"}
BASE_COLOR = "#9a9a9a"

# agrupa features estáticas por tema para a figura de importância
GROUPS = {
    "Força de quem joga hoje": ["LINEUP_STRENGTH", "STRENGTH_DIFF"],
    "Elenco completo": ["ROSTER_STRENGTH"],
    "Desfalques": ["MISSING_IMPACT", "MISSING_MIN", "STAR_OUT"],
    "Propensão a lesão": ["INJURY_RISK"],
    "Trocas (chegadas/saídas)": ["IMPACT_IN", "IMPACT_OUT", "CONTINUITY"],
    "Elo": ["ELO_PRE", "ELO_DIFF"],
    "Histórico de temporadas": ["HIST_WINPCT", "LAST_WINPCT"],
    "Calendário": ["REST", "B2B", "SEASON_PROGRESS"],
}


def load_model(cell: str, splits) -> TwoTowerNet:
    tr = splits["train"]
    model = TwoTowerNet(seq_dim=tr.seq_home.shape[-1], static_dim=tr.static.shape[-1],
                        units=config.UNITS, hidden=config.HIDDEN, dropout=config.DROPOUT, cell=cell)
    model.load_state_dict(torch.load(config.ARTIFACTS / f"{cell}.pt", map_location="cpu"))
    return model.eval()


def fig_comparison():
    df = pd.read_csv(config.ARTIFACTS / "results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in [(axes[0], "log_loss", "Log-loss (menor é melhor)"),
                              (axes[1], "accuracy", "Acurácia (maior é melhor)")]:
        colors = [NET_COLOR.get(m, BASE_COLOR) for m in df.modelo]
        err = df.get(f"{metric}_std")
        ax.bar(df.modelo, df[metric], color=colors,
               yerr=err.fillna(0) if err is not None else None, capsize=3)
        lo, hi = df[metric].min(), df[metric].max()
        pad = (hi - lo) * 0.6 + 0.005
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Teste: temporada {config.TEST_SEASON}")
    fig.tight_layout()
    fig.savefig(FIG / "comparacao_modelos.png", dpi=160)
    plt.close(fig)


def fig_curves():
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for cell, color in NET_COLOR.items():
        path = config.ARTIFACTS / f"history_{cell}.json"
        if not path.exists():
            continue
        h = json.loads(path.read_text())
        ax.plot([r["epoch"] for r in h], [r["log_loss"] for r in h], label=cell.upper(), color=color)
    ax.set_xlabel("Época")
    ax.set_ylabel("Log-loss na validação")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "curvas_treino.png", dpi=160)
    plt.close(fig)


def fig_calibration(test, p_cfc):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    bins = np.linspace(0, 1, 11)
    for name, p, color in [("CfC", p_cfc, NET_COLOR["cfc"]), ("Elo", test.elo_prob, BASE_COLOR)]:
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 15:
                xs.append(p[m].mean())
                ys.append(test.y[m].mean())
        ax.plot(xs, ys, "o-", label=name, color=color)
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Calibração perfeita")
    ax.set_xlabel("Probabilidade prevista de vitória do mandante")
    ax.set_ylabel("Frequência real")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "calibracao.png", dpi=160)
    plt.close(fig)


def permutation_importance(model, test, static_names, repeats=5):
    """Quanto a log-loss PIORA quando embaralhamos um grupo de features
    (nos dois times ao mesmo tempo). Maior = mais importante."""
    rng = np.random.default_rng(config.SEED)
    base = evaluate(test.y, predict(model, to_loader(test, False), "cpu"))["log_loss"]
    rows = []
    for group, feats in GROUPS.items():
        cols = [i for i, n in enumerate(static_names)
                if n.rsplit("_", 1)[0] in feats or n in feats]
        if not cols:
            continue
        deltas = []
        for _ in range(repeats):
            st = test.static.copy()
            perm = rng.permutation(len(st))
            st[:, cols] = st[perm][:, cols]
            shuffled = type(test)(**{**test.__dict__, "static": st})
            p = predict(model, to_loader(shuffled, False), "cpu")
            deltas.append(evaluate(test.y, p)["log_loss"] - base)
        rows.append({"grupo": group, "delta_logloss": np.mean(deltas), "std": np.std(deltas)})

    # a sequência de jogos anteriores (entrada da CfC) também como "grupo"
    deltas = []
    for _ in range(repeats):
        perm = rng.permutation(len(test))
        shuffled = type(test)(**{**test.__dict__, "seq_home": test.seq_home[perm],
                                 "ts_home": test.ts_home[perm], "seq_away": test.seq_away[perm],
                                 "ts_away": test.ts_away[perm]})
        p = predict(model, to_loader(shuffled, False), "cpu")
        deltas.append(evaluate(test.y, p)["log_loss"] - base)
    rows.append({"grupo": "Sequência dos últimos jogos (CfC)",
                 "delta_logloss": np.mean(deltas), "std": np.std(deltas)})
    return pd.DataFrame(rows).sort_values("delta_logloss")


def fig_importance(imp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(imp.grupo, imp.delta_logloss, xerr=imp["std"], color=NET_COLOR["cfc"], capsize=3)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("Aumento da log-loss ao embaralhar (maior = mais importante)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "importancia.png", dpi=160)
    plt.close(fig)


def fig_lineup_effect(model, test, static_names):
    """Varia a força de quem joga hoje no mandante (em desvios-padrão),
    mantendo o resto fixo, e mostra a probabilidade média prevista.
    Ex.: −1 dp ≈ perder um titular importante."""
    cols = [static_names.index(c) for c in ("LINEUP_STRENGTH_H", "STRENGTH_DIFF")
            if c in static_names]
    shifts = np.linspace(-2, 2, 9)
    probs = []
    for s in shifts:
        st = test.static.copy()
        st[:, cols] += s   # features já padronizadas: +1 = +1 desvio-padrão
        shifted = type(test)(**{**test.__dict__, "static": st})
        probs.append(predict(model, to_loader(shifted, False), "cpu").mean())
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(shifts, probs, "o-", color=NET_COLOR["cfc"])
    ax.axvline(0, color="#999", lw=1, ls="--")
    ax.set_xlabel("Força de quem joga pelo mandante (desvios-padrão em relação ao real)")
    ax.set_ylabel("P(mandante vence) média prevista")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "efeito_desfalque.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    splits, meta = load_splits()
    test = splits["test"]
    static_names = meta["static_features"]

    fig_comparison()
    fig_curves()
    model = load_model("cfc", splits)
    p = predict(model, to_loader(test, False), "cpu")
    fig_calibration(test, p)
    imp = permutation_importance(model, test, static_names)
    imp.to_csv(config.ARTIFACTS / "importancia.csv", index=False)
    fig_importance(imp)
    fig_lineup_effect(model, test, static_names)
    print(imp.round(4).to_string(index=False))
    print(f"\nFiguras salvas em {FIG}")


if __name__ == "__main__":
    main()
