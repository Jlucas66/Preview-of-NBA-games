"""Features da temporada EM ANDAMENTO, para o site prever os próximos jogos.

Uso (depois de `python -m src.data.collect --current`):
    python -m src.features.current

Usa exatamente o mesmo pipeline do treino (Elo, descanso, histórico, impacto
dos jogadores, desfalques, lesões, sequência dos últimos jogos). A diferença é
que os jogos futuros:
  * usam o ELENCO OFICIAL atual de cada time (CommonTeamRoster), e não os logs;
  * vêm com uma escalação PROVÁVEL: quem jogou em algum dos últimos
    PROBABLE_LAST_GAMES jogos do time. No site o usuário corrige marcando
    quem está lesionado;
  * não têm resultado, então não atualizam o Elo nem entram nas sequências.

Gera em data/processed/:
    current_games.csv, current.npz, current_rosters.parquet
"""
import numpy as np
import pandas as pd

from src import config
from src.features import build


def load_schedule(season: str) -> pd.DataFrame:
    path = config.DATA_RAW / f"schedule_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode: python -m src.data.collect --current")
    sc = pd.read_csv(path, dtype={"GAME_ID": str})
    sc["GAME_ID"] = sc["GAME_ID"].str.zfill(10)
    return sc


def probable_players(season: str, tg: pd.DataFrame, roster: pd.DataFrame) -> dict[int, set]:
    """Quem jogou em algum dos últimos N jogos do time (por time)."""
    pg = pd.read_csv(config.DATA_RAW / f"player_games_{season}.csv", dtype={"GAME_ID": str})
    pg["GAME_ID"] = pg["GAME_ID"].str.zfill(10)
    played = tg[(tg.SEASON == season) & (tg.FUTURE == 0)].sort_values("GAME_DATE")
    out = {}
    for team, r in roster.groupby("TEAM_ID"):
        last = played[played.TEAM_ID == team].GAME_ID.tail(config.PROBABLE_LAST_GAMES)
        if len(last):
            recent = set(pg[(pg.TEAM_ID == team) & pg.GAME_ID.isin(last)].PLAYER_ID)
            out[team] = set(r.PLAYER_ID) & recent
        else:                      # time ainda não jogou: todo o elenco
            out[team] = set(r.PLAYER_ID)
    return out


def main() -> None:
    season = config.CURRENT_SEASON
    sc = load_schedule(season)
    roster = pd.read_csv(config.DATA_RAW / f"rosters_{season}.csv")

    future = sc[sc.STATUS == 1].sort_values("GAME_DATE")
    if len(future):
        first = pd.Timestamp(future.GAME_DATE.iloc[0])
        future = future[pd.to_datetime(future.GAME_DATE) <= first + pd.Timedelta(days=config.UPCOMING_DAYS)]
    print(f"{len(future)} jogos agendados nos próximos {config.UPCOMING_DAYS} dias")

    seasons = config.team_seasons() + [season]
    tg = build.load_team_games(seasons, future=future)
    tg = build.add_elo(tg)
    tg = build.add_schedule_features(tg)
    tg = build.add_history_features(tg)

    probable = probable_players(season, tg, roster)
    official = roster.groupby("TEAM_ID")["PLAYER_ID"].apply(list).to_dict()
    overrides = {}
    for f in future.itertuples(index=False):
        for t in (f.HOME_TEAM_ID, f.AWAY_TEAM_ID):
            overrides[(f.GAME_ID, t)] = (official.get(t, []), probable.get(t, set()))

    # nomes do elenco oficial (jogadores que ainda não entraram em quadra não estão nos logs)
    names = dict(zip(roster.PLAYER_ID, roster.PLAYER_NAME))
    tg = build.add_roster_features(tg, seasons=[season], overrides=overrides,
                                   out_name="current_rosters.parquet")
    rp = config.DATA_PROCESSED / "current_rosters.parquet"
    r = pd.read_parquet(rp)
    r["PLAYER_NAME"] = [names.get(p, n) for p, n in zip(r.PLAYER_ID, r.PLAYER_NAME)]
    r.to_parquet(rp, index=False)

    games, data, _ = build.build_dataset(tg, seasons=[season])
    time_of = dict(zip(sc.GAME_ID, sc.GAME_TIME))
    games["GAME_TIME"] = games["GAME_ID"].map(time_of)
    games.to_csv(config.DATA_PROCESSED / "current_games.csv", index=False)
    np.savez_compressed(config.DATA_PROCESSED / "current.npz", **data)

    n_fut = int(games.FUTURE.sum())
    print(f"{len(games) - n_fut} jogos já disputados + {n_fut} futuros salvos em {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
