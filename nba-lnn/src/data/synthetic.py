"""Gera dados SINTÉTICOS no mesmo formato da nba_api e do hoopR.

Serve só para testar o pipeline sem internet. Os resultados de modelo com esses
dados não significam nada sobre a NBA real.

A simulação imita a estrutura real:
  * cada jogador tem um "impacto" verdadeiro (pontos de saldo por 100 posses);
  * a força do time no jogo = soma do impacto × minutos dos jogadores em quadra;
  * cada jogador tem uma propensão PRÓPRIA a se lesionar, que persiste entre
    temporadas; lesões duram vários jogos;
  * ~25% do elenco muda a cada offseason (trocas, free agency, draft);
  * mando de quadra e cansaço em back-to-back;
  * o "rating de impacto" publicado (como o do hoopR) é o impacto verdadeiro
    com ruído de medição.
"""
import numpy as np
import pandas as pd

from src import config

N_TEAMS = 30
ROSTER = 14
DAYS = 168


def _season_dates(season: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{config.season_start_year(season)}-10-22")
    return pd.date_range(start, periods=DAYS, freq="D")


CURRENT_PLAYED_DAYS = 24   # dias já "disputados" da temporada atual simulada


def generate(seed: int = config.SEED, current_only: bool = False) -> None:
    """Gera tudo (a simulação é determinística; current_only existe só por clareza)."""
    rng = np.random.default_rng(seed)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    team_ids = [1610612737 + i for i in range(N_TEAMS)]
    abbrs = {t: f"T{i:02d}" for i, t in enumerate(team_ids)}

    next_pid = 1
    impact: dict[int, float] = {}
    injury_rate: dict[int, float] = {}   # prob. de se lesionar em cada jogo
    rosters: dict[int, list[int]] = {}

    def new_player(mean: float, sd: float) -> int:
        nonlocal next_pid
        pid = next_pid
        next_pid += 1
        impact[pid] = rng.normal(mean, sd)
        injury_rate[pid] = rng.beta(1.2, 40)   # média ~3% por jogo, alguns bem "frágeis"
        return pid

    for t in team_ids:
        rosters[t] = [new_player(0, 2.5) for _ in range(ROSTER)]

    cur = config.CURRENT_SEASON
    seasons = sorted(set(config.team_seasons()) | set(config.player_seasons()) | {cur},
                     key=config.season_start_year)
    player_seasons = set(config.player_seasons()) | {cur}
    impact_seasons = set(config.impact_seasons()) | {
        config.season_from_year(config.season_start_year(cur) - 1)}

    for season in seasons:
        # ---- offseason: ~25% do elenco muda ----
        free_agents: list[int] = []
        for t in team_ids:
            n_out = rng.binomial(ROSTER, 0.25)
            out = list(rng.choice(rosters[t], size=n_out, replace=False))
            rosters[t] = [p for p in rosters[t] if p not in out]
            free_agents += out
        rng.shuffle(free_agents)
        for t in team_ids:
            while len(rosters[t]) < ROSTER:
                if free_agents and rng.random() < 0.7:
                    rosters[t].append(free_agents.pop())
                else:
                    rosters[t].append(new_player(-1.0, 2.0))   # novato
        for pid in impact:   # evolução / regressão
            impact[pid] = 0.9 * impact[pid] + rng.normal(0, 0.7)

        injured_until = {p: -1 for p in impact}   # índice do jogo do time até o qual está fora
        team_game_idx = {t: 0 for t in team_ids}
        team_rows, player_rows = [], []
        last_played = {t: None for t in team_ids}
        season_minutes: dict[int, float] = {}
        season_gp: dict[int, int] = {}
        gid = 0
        is_cur = season == cur
        dates = _season_dates(season)
        cutoff = dates[CURRENT_PLAYED_DAYS] if is_cur else None
        schedule_rows, roster_snapshot = [], None

        for day in dates:
            if is_cur and day == cutoff:
                roster_snapshot = {t: list(r) for t, r in rosters.items()}
            hidden = is_cur and day >= cutoff   # jogo ainda não disputado
            avail = [t for t in team_ids if team_game_idx[t] < 82]
            rng.shuffle(avail)
            n_games = min(len(avail) // 2, int(rng.integers(5, 10)))
            for g in range(n_games):
                home, away = avail[2 * g], avail[2 * g + 1]
                gid += 1
                game_id = f"002{config.season_start_year(season) % 100:02d}{gid:05d}"
                strengths, lineups = {}, {}
                for t in (home, away):
                    k = team_game_idx[t]
                    for p in rosters[t]:   # novas lesões
                        if injured_until[p] < k and rng.random() < injury_rate[p]:
                            injured_until[p] = k + int(rng.geometric(1 / 6))
                    healthy = [p for p in rosters[t] if injured_until[p] < k]
                    active = sorted(healthy, key=lambda p: -impact[p])[:10]
                    weights = np.linspace(1.6, 0.4, len(active))
                    mins = weights / weights.sum() * 240
                    lineups[t] = list(zip(active, mins))
                    s = sum(impact[p] * m / 48 for p, m in lineups[t])
                    if last_played[t] is not None and (day - last_played[t]).days == 1:
                        s -= 1.5
                    strengths[t] = s
                margin = strengths[home] - strengths[away] + 2.5 + rng.normal(0, 12)
                base = rng.normal(113, 7)
                pts = {home: int(round(base + margin / 2)), away: int(round(base - margin / 2))}
                if pts[home] == pts[away]:
                    pts[home] += 1
                if is_cur:
                    schedule_rows.append({
                        "GAME_ID": game_id, "GAME_DATE": day.strftime("%Y-%m-%d"),
                        "GAME_TIME": "7:30 pm ET", "STATUS": 1 if hidden else 3,
                        "HOME_TEAM_ID": home, "HOME_ABBR": abbrs[home],
                        "AWAY_TEAM_ID": away, "AWAY_ABBR": abbrs[away], "SEASON": season})

                for t, opp in ((home, away), (away, home)):
                    if hidden:
                        last_played[t] = day
                        team_game_idx[t] += 1
                        continue
                    fga = int(rng.normal(88, 5))
                    fgm = int(min(fga, max(25, pts[t] * 0.37 + rng.normal(0, 2))))
                    fg3a = int(rng.normal(35, 5))
                    fg3m = int(min(fg3a, max(5, rng.normal(13, 3))))
                    fta = int(max(5, rng.normal(22, 5)))
                    ftm = int(min(fta, fta * rng.uniform(0.65, 0.9)))
                    oreb, dreb = int(rng.normal(10, 3)), int(rng.normal(34, 4))
                    team_rows.append({
                        "SEASON": season, "TEAM_ID": t, "TEAM_ABBREVIATION": abbrs[t],
                        "GAME_ID": game_id, "GAME_DATE": day.strftime("%Y-%m-%d"),
                        "MATCHUP": f"{abbrs[t]} vs. {abbrs[opp]}" if t == home
                                   else f"{abbrs[t]} @ {abbrs[opp]}",
                        "WL": "W" if pts[t] > pts[opp] else "L",
                        "PTS": pts[t], "FGM": fgm, "FGA": fga, "FG_PCT": round(fgm / fga, 3),
                        "FG3M": fg3m, "FG3A": fg3a, "FG3_PCT": round(fg3m / max(fg3a, 1), 3),
                        "FTM": ftm, "FTA": fta, "FT_PCT": round(ftm / max(fta, 1), 3),
                        "OREB": oreb, "DREB": dreb, "REB": oreb + dreb,
                        "AST": int(rng.normal(26, 4)), "STL": int(rng.normal(8, 2)),
                        "BLK": int(rng.normal(5, 2)), "TOV": int(rng.normal(14, 3)),
                        "PF": int(rng.normal(20, 3)), "PLUS_MINUS": pts[t] - pts[opp],
                    })
                    for pid, m in lineups[t]:
                        m = max(1.0, float(m + rng.normal(0, 2)))
                        season_minutes[pid] = season_minutes.get(pid, 0) + m
                        season_gp[pid] = season_gp.get(pid, 0) + 1
                        if season in player_seasons:
                            player_rows.append({
                                "SEASON": season, "PLAYER_ID": pid, "PLAYER_NAME": f"Player {pid}",
                                "TEAM_ID": t, "GAME_ID": game_id,
                                "GAME_DATE": day.strftime("%Y-%m-%d"), "MIN": round(m, 1),
                            })
                    last_played[t] = day
                    team_game_idx[t] += 1

            # trocas no meio da temporada (raras)
            if rng.random() < 0.05:
                a, b = rng.choice(team_ids, size=2, replace=False)
                pa, pb = rng.choice(rosters[a]), rng.choice(rosters[b])
                rosters[a][rosters[a].index(pa)] = pb
                rosters[b][rosters[b].index(pb)] = pa

        if season in config.team_seasons() or is_cur:
            pd.DataFrame(team_rows).to_csv(config.DATA_RAW / f"team_games_{season}.csv", index=False)
        if is_cur:
            pd.DataFrame(schedule_rows).to_csv(config.DATA_RAW / f"schedule_{season}.csv", index=False)
            pd.DataFrame([{"TEAM_ID": t, "PLAYER_ID": p, "PLAYER_NAME": f"Player {p}",
                           "NUM": "", "POSITION": ""}
                          for t, ps in roster_snapshot.items() for p in ps]).to_csv(
                config.DATA_RAW / f"rosters_{season}.csv", index=False)
        if season in player_seasons:
            pd.DataFrame(player_rows).to_csv(config.DATA_RAW / f"player_games_{season}.csv", index=False)
        if season in impact_seasons:
            rows = []
            for pid, mins in season_minutes.items():
                noise = rng.normal(0, 1.5 * np.sqrt(500 / (mins + 500)))  # menos minutos = mais ruído
                r = impact[pid] + noise
                rows.append({"player_id": pid, "player_name": f"Player {pid}", "season": season,
                             "gp": season_gp[pid], "min": round(mins, 1),
                             "adj_rapm": r, "rapm": r + rng.normal(0, 0.5), "bpm": r + rng.normal(0, 2),
                             "spm": r / 3, "war": r * mins / 400})
            pd.DataFrame(rows).to_csv(config.DATA_RAW / f"player_impact_{season}.csv", index=False)
        print(f"[sintético] {season}: {len(team_rows) // 2} jogos")

    print("Dados sintéticos gerados em", config.DATA_RAW)
