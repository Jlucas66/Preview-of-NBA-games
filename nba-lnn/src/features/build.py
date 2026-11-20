"""Engenharia de features e montagem das sequências para a rede.

Uso:
    python -m src.features.build

Lê data/raw/ e gera:
    data/processed/games.csv     -> uma linha por jogo com as features estáticas (bom p/ EDA)
    data/processed/dataset.npz   -> tensores prontos para o treino

REGRA DE OURO (evitar vazamento): toda feature de um jogo usa apenas
informação disponível ANTES do jogo começar. A única exceção assumida é a
lista de jogadores que entram em quadra, que na prática é conhecida antes do
jogo pelo injury report / escalação divulgada.

Para cada jogo (mandante x visitante) temos:
  * seq_home / seq_away: os SEQ_LEN jogos anteriores de cada time
    (resultado, pontos, aproveitamento, Elo do adversário...) -> entrada da CfC
  * ts_home / ts_away: dias entre cada um desses jogos -> `timespans` da CfC
  * static: features do contexto do jogo (Elo, descanso, histórico das
    últimas temporadas e as features de elenco de src/features/players.py:
    força de quem joga, desfalques, reforços/saídas, propensão a lesão)
  * y: 1 se o mandante venceu
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config
from src.features import players

STEP_FEATURES = [
    "WIN", "PTS", "OPP_PTS", "MARGIN", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV",
    "OPP_FG_PCT", "OPP_FG3_PCT", "OPP_REB", "OPP_TOV", "HOME", "ELO_PRE", "OPP_ELO_PRE",
    "LINEUP_STRENGTH", "OPP_LINEUP_STRENGTH",   # quem jogou naquele jogo (dos dois lados)
]

# features estáticas por time (viram _H e _A) + features do jogo
TEAM_STATIC = ["ELO_PRE", "REST", "B2B", "HIST_WINPCT", "LAST_WINPCT", "SEASON_PROGRESS",
               *players.ROSTER_FEATURES]


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def _read_csvs(prefix: str, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = config.DATA_RAW / f"{prefix}_{s}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} não encontrado. Rode antes: python -m src.data.collect")
        frames.append(pd.read_csv(path, dtype={"GAME_ID": str}))
    df = pd.concat(frames, ignore_index=True)
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def load_team_games(seasons: list[str] | None = None,
                    future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Jogos já disputados (+ jogos futuros, sem resultado, se `future` for dado).

    future: colunas GAME_ID, GAME_DATE, SEASON, HOME_TEAM_ID, AWAY_TEAM_ID,
            HOME_ABBR, AWAY_ABBR. Esses jogos recebem FUTURE=1.
    """
    tg = _read_csvs("team_games", seasons or config.team_seasons())

    # mantém só jogos com exatamente 2 linhas (1 por time) e 1 mandante
    tg["HOME"] = tg["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    counts = tg.groupby("GAME_ID").agg(n=("TEAM_ID", "size"), h=("HOME", "sum"))
    valid = counts[(counts.n == 2) & (counts.h == 1)].index
    dropped = tg["GAME_ID"].nunique() - len(valid)
    if dropped:
        print(f"[aviso] {dropped} jogos descartados por dados incompletos")
    tg = tg[tg["GAME_ID"].isin(valid)].copy()

    opp = tg[["GAME_ID", "TEAM_ID", "PTS", "FG_PCT", "FG3_PCT", "REB", "TOV"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS", "FG_PCT": "OPP_FG_PCT",
                 "FG3_PCT": "OPP_FG3_PCT", "REB": "OPP_REB", "TOV": "OPP_TOV"})
    tg = tg.merge(opp, on="GAME_ID")
    tg = tg[tg["TEAM_ID"] != tg["OPP_TEAM_ID"]].copy()

    tg["WIN"] = (tg["PTS"] > tg["OPP_PTS"]).astype(float)
    tg["MARGIN"] = tg["PTS"] - tg["OPP_PTS"]
    tg["FUTURE"] = 0

    if future is not None and len(future):
        future = future[~future.GAME_ID.isin(tg.GAME_ID)]
        rows = []
        for f in future.itertuples(index=False):
            for team, opp, abbr, home in ((f.HOME_TEAM_ID, f.AWAY_TEAM_ID, f.HOME_ABBR, 1),
                                          (f.AWAY_TEAM_ID, f.HOME_TEAM_ID, f.AWAY_ABBR, 0)):
                rows.append({"SEASON": f.SEASON, "TEAM_ID": team, "OPP_TEAM_ID": opp,
                             "TEAM_ABBREVIATION": abbr, "GAME_ID": f.GAME_ID,
                             "GAME_DATE": pd.Timestamp(f.GAME_DATE), "HOME": home, "FUTURE": 1})
        tg = pd.concat([tg, pd.DataFrame(rows)], ignore_index=True)

    tg = tg.sort_values(["GAME_DATE", "GAME_ID", "HOME"]).reset_index(drop=True)
    return tg


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def add_elo(tg: pd.DataFrame) -> pd.DataFrame:
    """Elo pré-jogo de cada time (atualizado jogo a jogo, regride à média entre temporadas)."""
    games = (tg[tg.HOME == 1][["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "OPP_TEAM_ID", "MARGIN"]]
             .sort_values(["GAME_DATE", "GAME_ID"]))
    elo: dict[int, float] = defaultdict(lambda: config.ELO_START)
    current_season = None
    pre = {}
    for g in games.itertuples(index=False):
        if g.SEASON != current_season:
            for t in list(elo):
                elo[t] = config.ELO_SEASON_CARRY * elo[t] + (1 - config.ELO_SEASON_CARRY) * config.ELO_MEAN
            current_season = g.SEASON
        h, a = g.TEAM_ID, g.OPP_TEAM_ID
        eh, ea = elo[h], elo[a]
        pre[(g.GAME_ID, h)], pre[(g.GAME_ID, a)] = eh, ea
        if pd.isna(g.MARGIN):   # jogo futuro: não atualiza o Elo
            continue

        exp_h = 1 / (1 + 10 ** (-(eh + config.ELO_HOME_ADV - ea) / 400))
        res_h = 1.0 if g.MARGIN > 0 else 0.0
        diff_winner = (eh + config.ELO_HOME_ADV - ea) if res_h else (ea - eh - config.ELO_HOME_ADV)
        mult = ((abs(g.MARGIN) + 3) ** 0.8) / (7.5 + 0.006 * diff_winner)
        delta = config.ELO_K * mult * (res_h - exp_h)
        elo[h] += delta
        elo[a] -= delta

    tg["ELO_PRE"] = [pre[(gid, t)] for gid, t in zip(tg.GAME_ID, tg.TEAM_ID)]
    tg["OPP_ELO_PRE"] = [pre[(gid, t)] for gid, t in zip(tg.GAME_ID, tg.OPP_TEAM_ID)]
    return tg


# ---------------------------------------------------------------------------
# Descanso, progresso na temporada e histórico
# ---------------------------------------------------------------------------
def add_schedule_features(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    prev_date = tg.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    prev_season = tg.groupby("TEAM_ID")["SEASON"].shift(1)

    gap = (tg["GAME_DATE"] - prev_date).dt.days
    # GAP_DAYS: intervalo real (inclui o offseason) -> usado como timespan da CfC
    tg["GAP_DAYS"] = gap.fillna(config.MAX_REST_DAYS).astype(float)
    # REST: descanso antes do jogo, truncado; 1º jogo da temporada = descanso máximo
    rest = gap.where(prev_season == tg["SEASON"], config.MAX_REST_DAYS)
    tg["REST"] = rest.fillna(config.MAX_REST_DAYS).clip(1, config.MAX_REST_DAYS)
    tg["B2B"] = (tg["REST"] == 1).astype(int)
    tg["SEASON_PROGRESS"] = tg.groupby(["TEAM_ID", "SEASON"]).cumcount() / 82.0
    return tg.sort_values(["GAME_DATE", "GAME_ID", "HOME"]).reset_index(drop=True)


def add_history_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Aproveitamento nas últimas HISTORY_SEASONS temporadas (conhecido antes da temporada)."""
    wp = tg.groupby(["TEAM_ID", "SEASON"])["WIN"].mean().unstack()
    seasons = sorted(wp.columns, key=config.season_start_year)
    hist, last = {}, {}
    for i, s in enumerate(seasons):
        prev = seasons[max(0, i - config.HISTORY_SEASONS):i]
        for t in wp.index:
            vals = wp.loc[t, prev].dropna() if prev else pd.Series(dtype=float)
            hist[(t, s)] = float(vals.mean()) if len(vals) else 0.5
            last[(t, s)] = float(wp.loc[t, prev[-1]]) if prev and pd.notna(wp.loc[t, prev[-1]]) else 0.5
    tg["HIST_WINPCT"] = [hist[(t, s)] for t, s in zip(tg.TEAM_ID, tg.SEASON)]
    tg["LAST_WINPCT"] = [last[(t, s)] for t, s in zip(tg.TEAM_ID, tg.SEASON)]
    return tg


# ---------------------------------------------------------------------------
# Elenco (impacto dos jogadores, trocas, desfalques, lesões) -> src/features/players.py
# ---------------------------------------------------------------------------
def add_roster_features(tg: pd.DataFrame, seasons: list[str] | None = None,
                        overrides: dict | None = None,
                        out_name: str = "rosters.parquet") -> pd.DataFrame:
    feats, rosters = players.compute(tg, seasons=seasons, overrides=overrides)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    rosters.to_parquet(config.DATA_PROCESSED / out_name, index=False)  # usado pelo app
    tg = tg.join(feats, on=["GAME_ID", "TEAM_ID"])
    opp = tg[["GAME_ID", "TEAM_ID", "LINEUP_STRENGTH"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "LINEUP_STRENGTH": "OPP_LINEUP_STRENGTH"})
    return tg.merge(opp, on=["GAME_ID", "OPP_TEAM_ID"], how="left")


# ---------------------------------------------------------------------------
# Montagem do dataset
# ---------------------------------------------------------------------------
def build_dataset(tg: pd.DataFrame, seasons: list[str] | None = None):
    """Monta sequências + features estáticas para os jogos das `seasons`.

    A sequência de cada time usa só jogos JÁ DISPUTADOS (FUTURE=0); um jogo
    futuro enxerga os últimos SEQ_LEN jogos disputados antes dele.
    """
    seasons = seasons or config.MODEL_SEASONS
    L = config.SEQ_LEN
    F = len(STEP_FEATURES)

    # linha do tempo de cada time (todas as temporadas, inclusive histórico)
    tg = tg.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    # jogos antigos não têm features de elenco: preenche com a mediana do treino
    tg["LABEL"] = tg["WIN"]   # resultado real (NaN em jogos futuros), antes do preenchimento
    med = tg.loc[tg.SEASON.isin(config.train_seasons()), STEP_FEATURES].median()
    tg[STEP_FEATURES] = tg[STEP_FEATURES].fillna(med).fillna(0.0)
    timeline: dict[int, np.ndarray] = {}
    gaps: dict[int, np.ndarray] = {}
    position: dict[tuple, int] = {}
    for team, d in tg.groupby("TEAM_ID", sort=False):
        d = d.sort_values(["GAME_DATE", "GAME_ID"])
        played = (d["FUTURE"] == 0).to_numpy()
        timeline[team] = d.loc[played, STEP_FEATURES].to_numpy(dtype=np.float32)
        gaps[team] = d.loc[played, "GAP_DAYS"].to_numpy(dtype=np.float32)
        # posição = quantos jogos disputados o time tem antes deste
        n_before = np.concatenate([[0], np.cumsum(played)[:-1]])
        for i, gid in zip(n_before, d.GAME_ID):
            position[(gid, team)] = int(i)

    def seq_for(gid, team):
        i = position[(gid, team)]
        x = np.zeros((L, F), dtype=np.float32)
        ts = np.ones(L, dtype=np.float32)
        lo = max(0, i - L)
        n = i - lo
        if n:
            x[L - n:] = timeline[team][lo:i]
            ts[L - n:] = gaps[team][lo:i]
        return x, ts

    home = tg[(tg.HOME == 1) & tg.SEASON.isin(seasons)].copy()
    away = tg[tg.HOME == 0][["GAME_ID"] + TEAM_STATIC].set_index("GAME_ID")
    home = home.set_index("GAME_ID")
    static_h = home[TEAM_STATIC].add_suffix("_H")
    static_a = away.loc[home.index, TEAM_STATIC].add_suffix("_A")
    games = pd.concat([home[["SEASON", "GAME_DATE", "TEAM_ID", "OPP_TEAM_ID",
                             "TEAM_ABBREVIATION", "LABEL", "FUTURE"]], static_h, static_a], axis=1)
    games = games.rename(columns={"TEAM_ID": "HOME_TEAM_ID", "OPP_TEAM_ID": "AWAY_TEAM_ID",
                                  "TEAM_ABBREVIATION": "HOME_ABBR", "LABEL": "HOME_WIN"})
    games["ELO_DIFF"] = games["ELO_PRE_H"] - games["ELO_PRE_A"]
    games["STRENGTH_DIFF"] = games["LINEUP_STRENGTH_H"] - games["LINEUP_STRENGTH_A"]
    games["ELO_PROB_HOME"] = 1 / (1 + 10 ** (-(games["ELO_DIFF"] + config.ELO_HOME_ADV) / 400))
    games = games.sort_values(["GAME_DATE"]).reset_index()

    static_cols = [c + "_H" for c in TEAM_STATIC] + [c + "_A" for c in TEAM_STATIC] + ["ELO_DIFF", "STRENGTH_DIFF"]

    xs_h, ts_h, xs_a, ts_a = [], [], [], []
    for g in games.itertuples(index=False):
        x, t = seq_for(g.GAME_ID, g.HOME_TEAM_ID)
        xs_h.append(x); ts_h.append(t)
        x, t = seq_for(g.GAME_ID, g.AWAY_TEAM_ID)
        xs_a.append(x); ts_a.append(t)

    data = {
        "seq_home": np.stack(xs_h), "ts_home": np.stack(ts_h),
        "seq_away": np.stack(xs_a), "ts_away": np.stack(ts_a),
        "static": games[static_cols].to_numpy(dtype=np.float32),
        "y": games["HOME_WIN"].to_numpy(dtype=np.float32),
        "elo_prob": games["ELO_PROB_HOME"].to_numpy(dtype=np.float32),
        "season": games["SEASON"].to_numpy(dtype=str),
        "game_id": games["GAME_ID"].to_numpy(dtype=str),
    }
    meta = {"step_features": STEP_FEATURES, "static_features": static_cols, "seq_len": L}
    return games, data, meta


def main() -> None:
    print("Carregando jogos...")
    tg = load_team_games()
    print("Calculando Elo, descanso e histórico...")
    tg = add_elo(tg)
    tg = add_schedule_features(tg)
    tg = add_history_features(tg)
    print("Calculando features de elenco (impacto, trocas, desfalques, lesões)...")
    tg = add_roster_features(tg)

    games, data, meta = build_dataset(tg)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    games.to_csv(config.DATA_PROCESSED / "games.csv", index=False)
    np.savez_compressed(config.DATA_PROCESSED / "dataset.npz", **data)
    (config.DATA_PROCESSED / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{len(games)} jogos | seq {data['seq_home'].shape} | static {data['static'].shape}")
    print(games.groupby("SEASON")["HOME_WIN"].agg(jogos="size", vitoria_mandante="mean").round(3))
    n_nan = int(np.isnan(data["static"]).sum())
    if n_nan:
        print(f"[aviso] {n_nan} valores ausentes nas features estáticas (serão preenchidos no treino)")


if __name__ == "__main__":
    main()
