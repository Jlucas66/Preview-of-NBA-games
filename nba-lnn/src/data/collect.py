"""Coleta dos dados brutos da NBA via nba_api (stats.nba.com).

Uso (a partir da raiz do projeto):
    python -m src.data.collect              # baixa dados reais
    python -m src.data.collect --synthetic  # gera dados falsos com o mesmo formato (teste offline)
    python -m src.data.collect --current    # temporada em andamento (próximos jogos no site)

Gera em data/raw/:
    team_games_<temporada>.csv     -> uma linha por (time, jogo)          [nba_api]
    player_games_<temporada>.csv   -> uma linha por (jogador, jogo)       [nba_api]
    player_impact_<temporada>.csv  -> uma linha por (jogador, temporada)  [hoopR / GitHub]

Os arquivos que já existem são pulados, então dá para rodar de novo se a
conexão cair no meio.
"""
import argparse
import time

import pandas as pd

from src import config

TEAM_COLS = [
    "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]
PLAYER_COLS = ["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN"]

# O stats.nba.com às vezes bloqueia requisições sem cabeçalhos de navegador.
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


def _call(endpoint_cls, retries: int = 4, **kwargs) -> pd.DataFrame:
    """Chama um endpoint com algumas tentativas e espera crescente."""
    for attempt in range(1, retries + 1):
        try:
            resp = endpoint_cls(headers=HEADERS, timeout=90, **kwargs)
            return resp.get_data_frames()[0]
        except Exception as exc:  # noqa: BLE001 - qualquer falha de rede
            wait = 10 * attempt
            print(f"   tentativa {attempt} falhou ({exc.__class__.__name__}: {exc}); "
                  f"esperando {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Falha ao baixar {endpoint_cls.__name__} com {kwargs}")


def fetch_team_games(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    df = _call(
        leaguegamefinder.LeagueGameFinder,
        player_or_team_abbreviation="T",
        league_id_nullable="00",
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    df["SEASON"] = season
    return df[TEAM_COLS]


def fetch_player_games(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs

    df = _call(
        playergamelogs.PlayerGameLogs,
        league_id_nullable="00",
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    df["SEASON"] = season
    return df[PLAYER_COLS]


IMPACT_COLS = ["player_id", "player_name", "season", "gp", "min",
               "adj_rapm", "rapm", "bpm", "spm", "war"]


def fetch_player_impact(season: str) -> pd.DataFrame:
    """Ratings de impacto (RAPM, adj-RAPM, BPM, SPM, WAR) do SportsDataverse/hoopR.

    Os arquivos ficam nas releases do GitHub sportsdataverse/sportsdataverse-data,
    um por temporada (identificada pelo ano em que TERMINA). `player_id` é o
    mesmo id de jogador do stats.nba.com, então junta direto com os logs.
    """
    url = config.IMPACT_URL.format(year=config.hoopr_year(season))
    df = pd.read_parquet(url)
    df = df[df["season_type"] == "Regular Season"].copy()
    df["season"] = season
    return df[IMPACT_COLS]


def fetch_schedule(season: str) -> pd.DataFrame:
    """Agenda completa da temporada (jogos disputados e futuros)."""
    from nba_api.stats.endpoints import scheduleleaguev2

    df = _call(scheduleleaguev2.ScheduleLeagueV2, league_id="00", season=season)
    df = df[df["gameId"].astype(str).str.zfill(10).str.startswith("002")]  # temporada regular
    return pd.DataFrame({
        "GAME_ID": df["gameId"].astype(str).str.zfill(10),
        "GAME_DATE": pd.to_datetime(df["gameDateEst"]).dt.strftime("%Y-%m-%d"),
        "GAME_TIME": df["gameStatusText"],
        "STATUS": df["gameStatus"],          # 1 = agendado, 2 = em andamento, 3 = encerrado
        "HOME_TEAM_ID": df["homeTeam_teamId"], "HOME_ABBR": df["homeTeam_teamTricode"],
        "AWAY_TEAM_ID": df["awayTeam_teamId"], "AWAY_ABBR": df["awayTeam_teamTricode"],
        "SEASON": season,
    })


def fetch_rosters(season: str, pause: float = 1.0) -> pd.DataFrame:
    """Elenco OFICIAL atual de cada time (inclui quem ainda não jogou, ex.: lesionados)."""
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams

    frames = []
    for t in teams.get_teams():
        df = _call(commonteamroster.CommonTeamRoster, team_id=t["id"], season=season)
        frames.append(pd.DataFrame({"TEAM_ID": t["id"], "PLAYER_ID": df["PLAYER_ID"],
                                    "PLAYER_NAME": df["PLAYER"], "NUM": df["NUM"],
                                    "POSITION": df["POSITION"]}))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True)


def collect_current(pause: float = 2.0) -> None:
    """Dados da temporada EM ANDAMENTO (rode de novo sempre que quiser atualizar)."""
    season = config.CURRENT_SEASON
    prev = config.season_from_year(config.season_start_year(season) - 1)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    path = config.DATA_RAW / f"player_impact_{prev}.csv"
    if not path.exists():
        print(f"[..] baixando impacto dos jogadores {prev} (hoopR)")
        fetch_player_impact(prev).to_csv(path, index=False)
    print(f"[..] agenda {season}")
    fetch_schedule(season).to_csv(config.DATA_RAW / f"schedule_{season}.csv", index=False)
    time.sleep(pause)
    print(f"[..] jogos já disputados {season}")
    fetch_team_games(season).to_csv(config.DATA_RAW / f"team_games_{season}.csv", index=False)
    time.sleep(pause)
    print(f"[..] logs de jogadores {season}")
    fetch_player_games(season).to_csv(config.DATA_RAW / f"player_games_{season}.csv", index=False)
    time.sleep(pause)
    print(f"[..] elencos oficiais {season} (30 times)")
    fetch_rosters(season).to_csv(config.DATA_RAW / f"rosters_{season}.csv", index=False)
    print("Dados da temporada atual atualizados. Agora: python -m src.features.current")


def collect_real(pause: float = 2.0) -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    for season in config.impact_seasons():
        path = config.DATA_RAW / f"player_impact_{season}.csv"
        if path.exists():
            print(f"[ok] {path.name} já existe")
            continue
        print(f"[..] baixando impacto dos jogadores {season} (hoopR)")
        fetch_player_impact(season).to_csv(path, index=False)

    for season in config.team_seasons():
        path = config.DATA_RAW / f"team_games_{season}.csv"
        if path.exists():
            print(f"[ok] {path.name} já existe")
            continue
        print(f"[..] baixando jogos dos times {season}")
        fetch_team_games(season).to_csv(path, index=False)
        time.sleep(pause)

    for season in config.player_seasons():
        path = config.DATA_RAW / f"player_games_{season}.csv"
        if path.exists():
            print(f"[ok] {path.name} já existe")
            continue
        print(f"[..] baixando logs de jogadores {season}")
        fetch_player_games(season).to_csv(path, index=False)
        time.sleep(pause)

    print("Coleta concluída.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true",
                        help="gera dados sintéticos em vez de baixar (teste offline)")
    parser.add_argument("--current", action="store_true",
                        help=f"baixa a temporada em andamento ({config.CURRENT_SEASON}) para o site")
    parser.add_argument("--impact-only", action="store_true",
                        help="baixa só os ratings de impacto (GitHub), sem chamar o stats.nba.com")
    args = parser.parse_args()

    if args.current and args.synthetic:
        from src.data.synthetic import generate
        generate(current_only=True)
    elif args.current:
        collect_current()
    elif args.impact_only:
        config.DATA_RAW.mkdir(parents=True, exist_ok=True)
        for season in config.impact_seasons():
            df = fetch_player_impact(season)
            df.to_csv(config.DATA_RAW / f"player_impact_{season}.csv", index=False)
            print(f"[ok] impacto {season}: {len(df)} jogadores")
    elif args.synthetic:
        from src.data.synthetic import generate
        generate()
    else:
        collect_real()


if __name__ == "__main__":
    main()
