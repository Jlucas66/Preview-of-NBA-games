"""Treino e avaliação.

Uso:
    python -m src.train                   # treina a CfC (modelo principal)
    python -m src.train --model ltc       # outro encoder: cfc | ncp | ltc | lstm
    python -m src.train --model all       # baselines + todos os encoders, tabela comparativa
    python -m src.train --model all --seeds 5   # idem, 5 sementes por rede (use no artigo)

Divisão temporal (definida em src/config.py):
    treino = temporadas antigas | validação = early stopping | teste = última temporada
Nada é embaralhado entre temporadas: o modelo nunca vê o futuro.

Saídas em artifacts/:
    <modelo>.pt                 pesos do melhor epoch (pela log-loss de validação)
    preds_<modelo>.csv          probabilidades previstas no teste
    results.csv                 tabela comparativa (com --model all)
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.models.baselines import run_baselines
from src.models.dataset import Split, load_splits
from src.models.metrics import evaluate
from src.models.networks import TwoTowerNet

ENCODERS = ["cfc", "ncp", "ltc", "lstm"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_loader(s: Split, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(a) for a in
               (s.seq_home, s.ts_home, s.seq_away, s.ts_away, s.static, s.y.astype(np.float32))]
    return DataLoader(TensorDataset(*tensors), batch_size=config.BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for xh, th, xa, ta, st, _ in loader:
        logits = model(xh.to(device), th.to(device), xa.to(device), ta.to(device), st.to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_model(cell: str, splits: dict, epochs: int, device: str, verbose: bool = True,
                seed: int = config.SEED):
    set_seed(seed)
    tr, va = splits["train"], splits["val"]
    model = TwoTowerNet(
        seq_dim=tr.seq_home.shape[-1], static_dim=tr.static.shape[-1],
        units=config.UNITS, hidden=config.HIDDEN, dropout=config.DROPOUT, cell=cell,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loader, val_loader = to_loader(tr, True), to_loader(va, False)

    best_loss, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xh, th, xa, ta, st, y in train_loader:
            xh, th, xa, ta, st, y = (t.to(device) for t in (xh, th, xa, ta, st, y))
            opt.zero_grad()
            loss = loss_fn(model(xh, th, xa, ta, st), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)

        val_metrics = evaluate(va.y, predict(model, val_loader, device))
        history.append({"epoch": epoch, "train_loss": total / len(tr), **val_metrics})
        if verbose:
            print(f"[{cell}] epoch {epoch:02d} | treino {total / len(tr):.4f} | "
                  f"val logloss {val_metrics['log_loss']:.4f} acc {val_metrics['accuracy']:.3f}")

        if val_metrics["log_loss"] < best_loss - 1e-4:
            best_loss, bad = val_metrics["log_loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.PATIENCE:
                if verbose:
                    print(f"[{cell}] early stopping no epoch {epoch}")
                break

    model.load_state_dict(best_state)
    info = {"params": n_params, "seconds": round(time.time() - t0, 1), "history": history}
    return model, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="cfc", choices=ENCODERS + ["all"])
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seeds", type=int, default=1,
                        help="treina cada rede N vezes com sementes diferentes e reporta média ± desvio")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, meta = load_splits()
    print(f"Dispositivo: {device} | treino {len(splits['train'])} | "
          f"val {len(splits['val'])} | teste {len(splits['test'])} jogos")
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    test = splits["test"]
    test_loader = to_loader(test, False)

    results = []
    if args.model == "all":
        for name, p in run_baselines(splits["train"], test).items():
            results.append({"modelo": name, **evaluate(test.y, p), "params": 0, "seconds": 0})

    encoders = ENCODERS if args.model == "all" else [args.model]
    for cell in encoders:
        best = None
        for i in range(args.seeds):
            seed = config.SEED + i
            model, info = train_model(cell, splits, args.epochs, device,
                                      verbose=not args.quiet, seed=seed)
            p = predict(model, test_loader, device)
            m = evaluate(test.y, p)
            results.append({"modelo": cell, "seed": seed, **m,
                            "params": info["params"], "seconds": info["seconds"]})
            # guarda o modelo da semente com melhor log-loss de VALIDAÇÃO
            val_ll = min(h["log_loss"] for h in info["history"])
            if best is None or val_ll < best[0]:
                best = (val_ll, model, p, info)

        _, model, p, info = best
        torch.save(model.state_dict(), config.ARTIFACTS / f"{cell}.pt")
        pd.DataFrame({"GAME_ID": test.game_id, "y": test.y, "p_home": p}).to_csv(
            config.ARTIFACTS / f"preds_{cell}.csv", index=False)
        (config.ARTIFACTS / f"history_{cell}.json").write_text(json.dumps(info["history"], indent=2))

    df = pd.DataFrame(results)
    metrics = ["accuracy", "log_loss", "brier", "auc"]
    summary = df.groupby("modelo", sort=False).agg(
        **{f"{m}": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        params=("params", "first"), seconds=("seconds", "mean"), runs=("modelo", "size"),
    ).reset_index()
    print(f"\n=== Resultados no TESTE ({config.TEST_SEASON}) ===")
    cols = ["modelo"] + metrics + (["log_loss_std", "accuracy_std"] if args.seeds > 1 else []) \
        + ["params", "seconds"]
    print(summary[cols].round(4).to_string(index=False))
    if args.model == "all":
        summary.to_csv(config.ARTIFACTS / "results.csv", index=False)
        df.to_csv(config.ARTIFACTS / "results_por_semente.csv", index=False)
        print(f"\nTabela salva em {config.ARTIFACTS / 'results.csv'}")

if __name__ == "__main__":
    main()
