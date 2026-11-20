"""Features de ELENCO baseadas no impacto de cada jogador.

Ideia central: a força de um time num jogo depende de QUEM está em quadra.
Cada jogador tem um peso (rating de impacto da temporada anterior, ex.:
adj-RAPM do hoopR = pontos de saldo por 100 posses que ele adiciona) e um
tempo de quadra esperado. A partir disso, para cada (time, jogo):

  LINEUP_STRENGTH  força de quem joga HOJE: média do rating ponderada pelos
                   minutos esperados, ×5 (≈ saldo por 100 posses do quinteto)
  ROSTER_STRENGTH  mesma conta com o elenco inteiro saudável (teto do time)
  MISSING_IMPACT   impacto somado dos jogadores do elenco que NÃO jogam hoje
                   (desfalques por lesão, descanso, suspensão)
  MISSING_MIN      fração dos minutos esperados do elenco que está fora hoje
  STAR_OUT         1 se o jogador mais importante do elenco está fora
  INJURY_RISK      propensão a lesão do elenco: % de jogos perdidos nas
                   últimas INJURY_SEASONS temporadas, ponderada por minutos
  IMPACT_IN        impacto dos reforços (não estavam no time na temporada passada)
  IMPACT_OUT       impacto de quem saiu (estava no time e não está mais)
  CONTINUITY       % dos minutos da temporada passada que continuam no elenco

Sem vazamento: ratings e histórico de lesões vêm de temporadas ANTERIORES;
minutos esperados vêm de jogos ANTERIORES. A única informação do próprio dia
é a lista de quem entra em quadra, que na NBA é conhecida antes do jogo pelo
injury report oficial.

Limitação assumida: os logs do stats.nba.com só registram quem JOGOU. "Jogo
perdido" = o time jogou e o jogador (que estava no elenco) não entrou; isso
mistura lesão com descanso e opção técnica (DNP-CD). Para o histórico usamos
isso como proxy de propensão a lesão.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src import config

ROSTER_FEATURES = ["LINEUP_STRENGTH", "ROSTER_STRENGTH", "MISSING_IMPACT", "MISSING_MIN",
                   "STAR_OUT", "INJURY_RISK", "IMPACT_IN", "IMPACT_OUT", "CONTINUITY"]
DEFAULT_MINUTES = 12.0


def _minutes_to_float(col: pd.Series) -> pd.Series:
    """Alguns endpoints devolvem minutos como '34:21'."""
    def conv(v):
        if isinstance(v, str) and ":" in v:
            m, s = v.split(":")[:2]
            return float(m) + float(s) / 60
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    return col.map(conv)


def load_player_games(seasons: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for s in seasons or config.player_seasons():
        path = config.DATA_RAW / f"player_games_{s}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} não encontrado. Rode: python -m src.data.collect")
        frames.append(pd.read_csv(path, dtype={"GAME_ID": str}))
    pg = pd.concat(frames, ignore_index=True)
    pg["GAME_ID"] = pg["GAME_ID"].astype(str).str.zfill(10)
    pg["GAME_DATE"] = pd.to_datetime(pg["GAME_DATE"])
    pg["MIN"] = _minutes_to_float(pg["MIN"])
    return pg[pg["MIN"] > 0].copy()


def load_ratings(seasons: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """season -> DataFrame indexado por player_id com rating e minutos por jogo."""
    out = {}
    for s in seasons or config.impact_seasons():
        path = config.DATA_RAW / f"player_impact_{s}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} não encontrado. Rode: python -m src.data.collect")
        d = pd.read_csv(path)
        d = d.groupby("player_id").agg(rating=(config.IMPACT_METRIC, "mean"),
                                       min=("min", "sum"), gp=("gp", "sum"))
        d["mpg"] = d["min"] / d["gp"].clip(lower=1)
        out[s] = d
    return out


# ---------------------------------------------------------------------------
# Histórico de jogos perdidos (proxy de propensão a lesão)
# ---------------------------------------------------------------------------
def games_missed_by_season(pg: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Para cada (temporada, jogador): jogos possíveis e jogos perdidos.

    Um jogador "pertence" a um time do seu 1º ao último jogo por ele na
    temporada. O período é estendido até o fim da temporada se foi o último time
    dele (pega lesões que encerram a temporada) e até o início se ele já estava
    no time na temporada anterior (pega quem começou a temporada lesionado).
    """
    team_games = (tg[["SEASON", "TEAM_ID", "GAME_ID", "GAME_DATE"]]
                  .drop_duplicates()
                  .sort_values(["SEASON", "TEAM_ID", "GAME_DATE", "GAME_ID"]))
    team_games["IDX"] = team_games.groupby(["SEASON", "TEAM_ID"]).cumcount()
    n_team_games = team_games.groupby(["SEASON", "TEAM_ID"])["IDX"].max() + 1
    idx_of = dict(zip(zip(team_games.SEASON, team_games.TEAM_ID, team_games.GAME_ID),
                      team_games.IDX))

    pg = pg.copy()
    pg["IDX"] = [idx_of.get(k, np.nan) for k in zip(pg.SEASON, pg.TEAM_ID, pg.GAME_ID)]
    pg = pg.dropna(subset=["IDX"])

    stints = (pg.groupby(["SEASON", "PLAYER_ID", "TEAM_ID"])
                .agg(first=("IDX", "min"), last=("IDX", "max"), played=("IDX", "nunique"),
                     first_date=("GAME_DATE", "min"))
                .reset_index()
                .sort_values(["SEASON", "PLAYER_ID", "first_date"]))

    # quem estava em cada time na temporada anterior
    prev_members = set()
    seasons = sorted(stints.SEASON.unique(), key=config.season_start_year)
    team_of_prev = {}
    for i, s in enumerate(seasons):
        last_stint = stints[stints.SEASON == s].groupby("PLAYER_ID").tail(1)
        team_of_prev[s] = dict(zip(last_stint.PLAYER_ID, last_stint.TEAM_ID))
    for i, s in enumerate(seasons[1:], start=1):
        for p, t in team_of_prev[seasons[i - 1]].items():
            prev_members.add((s, p, t))

    rows = []
    for (s, p), d in stints.groupby(["SEASON", "PLAYER_ID"], sort=False):
        possible = played = 0
        for j, r in enumerate(d.itertuples(index=False)):
            first, last = r.first, r.last
            if j == len(d) - 1:                          # último time na temporada
                last = n_team_games[(s, r.TEAM_ID)] - 1
            if j == 0 and (s, p, r.TEAM_ID) in prev_members:  # continuou no time
                first = 0
            possible += last - first + 1
            played += r.played
        rows.append((s, p, possible, played))
    out = pd.DataFrame(rows, columns=["SEASON", "PLAYER_ID", "POSSIBLE", "PLAYED"])
    out["MISSED"] = (out["POSSIBLE"] - out["PLAYED"]).clip(lower=0)
    return out


def injury_rates(missed: pd.DataFrame, season: str) -> tuple[dict[int, float], float]:
    """Taxa de jogos perdidos nas INJURY_SEASONS anteriores a `season`, com
    suavização bayesiana em direção à média da liga (peso = 1 temporada)."""
    y = config.season_start_year(season)
    prev = [config.season_from_year(y - k) for k in range(1, config.INJURY_SEASONS + 1)]
    m = missed[missed.SEASON.isin(prev)].groupby("PLAYER_ID")[["MISSED", "POSSIBLE"]].sum()
    league = m["MISSED"].sum() / max(m["POSSIBLE"].sum(), 1)
    k = 82.0
    rates = ((m["MISSED"] + k * league) / (m["POSSIBLE"] + k)).to_dict()
    return rates, float(league)


# ---------------------------------------------------------------------------
# Features jogo a jogo
# ---------------------------------------------------------------------------
def lineup_features(rating, exp_min, injury, plays) -> dict:
    """Features que dependem de QUEM JOGA, a partir dos jogadores do elenco.

    Recebe arrays alinhados (um item por jogador do elenco): rating, minutos
    esperados, taxa de jogos perdidos e se joga hoje (bool). Usada tanto no
    treino quanto no app (simulador de desfalques), garantindo a mesma conta.
    """
    rating, em = np.asarray(rating, float), np.asarray(exp_min, float)
    injury, plays = np.asarray(injury, float), np.asarray(plays, bool)
    if len(rating) == 0:
        return {"LINEUP_STRENGTH": 0.0, "ROSTER_STRENGTH": 0.0, "MISSING_IMPACT": 0.0,
                "MISSING_MIN": 0.0, "STAR_OUT": 0.0, "INJURY_RISK": 0.0}
    tot = em.sum() or 1.0
    play_min = em[plays].sum() or 1.0
    contrib = rating * em
    return {
        "LINEUP_STRENGTH": 5 * contrib[plays].sum() / play_min,
        "ROSTER_STRENGTH": 5 * contrib.sum() / tot,
        "MISSING_IMPACT": (contrib[~plays] / 48).sum(),
        "MISSING_MIN": em[~plays].sum() / tot,
        "STAR_OUT": float(not plays[int(np.argmax(contrib))]),
        "INJURY_RISK": (injury * em).sum() / tot,
    }


def compute(tg: pd.DataFrame, seasons: list[str] | None = None,
            overrides: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (features, elencos).

    features: ROSTER_FEATURES indexado por (GAME_ID, TEAM_ID)
    elencos:  uma linha por (jogo, time, jogador do elenco) com rating, minutos
              esperados, taxa de jogos perdidos e se jogou -> usado pelo app
    seasons:  temporadas para as quais calcular (padrão: MODEL_SEASONS)
    overrides: {(GAME_ID, TEAM_ID): (elenco, quem_joga)} para jogos FUTUROS:
              o elenco oficial e a escalação provável substituem os logs.
    """
    seasons = seasons or config.MODEL_SEASONS
    overrides = overrides or {}
    first = config.season_start_year(min(seasons, key=config.season_start_year))
    last = config.season_start_year(max(seasons, key=config.season_start_year))
    pg = load_player_games([config.season_from_year(y)
                            for y in range(first - config.INJURY_SEASONS, last + 1)])
    ratings = load_ratings([config.season_from_year(config.season_start_year(s) - 1)
                            for s in seasons])
    if "FUTURE" in tg:
        tg_played = tg[tg.FUTURE == 0]
    else:
        tg_played = tg
    names = pg.sort_values("GAME_DATE").groupby("PLAYER_ID")["PLAYER_NAME"].last().to_dict()
    roster_rows = []
    missed = games_missed_by_season(pg, tg_played)

    lineup = pg.groupby(["GAME_ID", "TEAM_ID"]).apply(
        lambda d: dict(zip(d.PLAYER_ID, d.MIN)), include_groups=False).to_dict()

    feats = {}
    for season in sorted(seasons, key=config.season_start_year):
        prev = config.season_from_year(config.season_start_year(season) - 1)
        rt = ratings[prev]
        rating = rt["rating"].to_dict()
        prior_mpg = rt["mpg"].to_dict()
        inj, league_inj = injury_rates(missed, season)

        # elenco e minutos de cada time na temporada anterior
        pprev = pg[pg.SEASON == prev]
        prev_team_min = pprev.groupby(["TEAM_ID", "PLAYER_ID"])["MIN"].sum()
        prev_team_games = pprev.groupby("TEAM_ID")["GAME_ID"].nunique()

        def r(p):
            return rating.get(p, config.ROOKIE_RATING)

        last_team: dict[int, int] = {}
        last_idx: dict[int, int] = {}
        recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.MINUTES_WINDOW))
        team_idx: dict[int, int] = defaultdict(int)
        members: dict[int, set] = defaultdict(set)

        def exp_min(p):
            if recent[p]:
                return float(np.mean(recent[p]))
            return float(prior_mpg.get(p, DEFAULT_MINUTES))

        games = (tg[tg.SEASON == season][["GAME_ID", "TEAM_ID", "GAME_DATE"]]
                 .sort_values(["GAME_DATE", "GAME_ID"]))
        for gid, gdf in games.groupby("GAME_ID", sort=False):
            todays = {}
            for t in gdf.TEAM_ID:
                k = team_idx[t]
                if (gid, t) in overrides:          # jogo futuro: elenco oficial
                    official, probable = overrides[(gid, t)]
                    roster = set(official)
                    today = {p: None for p in probable if p in roster}
                else:
                    today = lineup.get((gid, t), {})
                    roster = {p for p in members[t]
                              if last_team.get(p) == t and k - last_idx[p] <= config.ROSTER_MEMORY_GAMES}
                roster |= set(today)
                plist = sorted(roster)
                em = {p: exp_min(p) for p in plist}
                lf = lineup_features([r(p) for p in plist], [em[p] for p in plist],
                                     [inj.get(p, league_inj) for p in plist],
                                     [p in today for p in plist])
                for p in plist:
                    roster_rows.append((gid, t, p, names.get(p, str(p)), r(p), em[p],
                                        inj.get(p, league_inj), p in today))

                prev_members = (prev_team_min.loc[t].to_dict()
                                if t in prev_team_min.index.get_level_values(0) else {})
                n_prev = float(prev_team_games.get(t, 82))
                total_prev = sum(prev_members.values()) or np.nan
                impact_in = sum(r(p) * em[p] / 48 for p in roster if p not in prev_members)
                impact_out = sum(r(p) * (m / n_prev) / 48
                                 for p, m in prev_members.items() if p not in roster)
                continuity = sum(m for p, m in prev_members.items() if p in roster) / total_prev

                feats[(gid, t)] = (lf["LINEUP_STRENGTH"], lf["ROSTER_STRENGTH"],
                                   lf["MISSING_IMPACT"], lf["MISSING_MIN"], lf["STAR_OUT"],
                                   lf["INJURY_RISK"], impact_in, impact_out, continuity)
                todays[t] = today

            # atualiza o estado DEPOIS de calcular as features dos dois times
            for t, today in todays.items():
                if (gid, t) in overrides:   # jogo futuro não muda o estado
                    continue
                for p, m in today.items():
                    last_team[p] = t
                    last_idx[p] = team_idx[t]
                    recent[p].append(m)
                    members[t].add(p)
                team_idx[t] += 1

    idx = pd.MultiIndex.from_tuples(list(feats.keys()), names=["GAME_ID", "TEAM_ID"])
    features = pd.DataFrame(list(feats.values()), index=idx, columns=ROSTER_FEATURES)
    rosters = pd.DataFrame(roster_rows, columns=["GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME",
                                                 "RATING", "EXP_MIN", "INJURY_RATE", "PLAYED"])
    return features, rosters
