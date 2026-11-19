"""Configurações centrais do projeto.

Altere aqui as temporadas, o tamanho da sequência e os hiperparâmetros.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Temporadas
# ---------------------------------------------------------------------------
# Temporadas em que o modelo é treinado/validado/testado (temporada regular).
MODEL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"
# Treino = MODEL_SEASONS menos validação e teste.

# Temporada em andamento: usada só para PREVER os próximos jogos no site
# (não entra no treino). Baixe com: python -m src.data.collect --current
CURRENT_SEASON = "2026-27"
UPCOMING_DAYS = 14      # quantos dias à frente mostrar no site
PROBABLE_LAST_GAMES = 3  # "provável" = jogou em algum dos últimos N jogos do time

# Quantas temporadas anteriores usar no "histórico do time".
HISTORY_SEASONS = 5

# Quantas temporadas anteriores usar no histórico de jogos perdidos de cada
# jogador (proxy de propensão a lesão).
INJURY_SEASONS = 3

# ---------------------------------------------------------------------------
# Impacto dos jogadores (SportsDataverse / hoopR Player Impact Model)
# ---------------------------------------------------------------------------
# Cada jogador recebe o rating da TEMPORADA ANTERIOR (já fechado antes do
# jogo -> sem vazamento). Opções: adj_rapm | rapm | bpm | spm
IMPACT_METRIC = "adj_rapm"
# Rating de quem não tem temporada anterior (novatos): ~ percentil 25 da liga
ROOKIE_RATING = -1.0
IMPACT_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
              "nba_player_impact/nba_player_impact_{year}.parquet")
# Um jogador conta como "do elenco" se jogou pelo time nos últimos N jogos do time
ROSTER_MEMORY_GAMES = 15
# Média de minutos esperados: últimos N jogos do jogador
MINUTES_WINDOW = 10

# ---------------------------------------------------------------------------
# Features / sequências
# ---------------------------------------------------------------------------
SEQ_LEN = 10            # quantos jogos anteriores de cada time entram na rede
MISSING_WINDOW = 10     # janela (jogos) para medir a importância de cada jogador
MAX_REST_DAYS = 7       # descanso acima disso é truncado (ex.: 1º jogo da temporada)

# Elo (mesmos parâmetros usados pelo FiveThirtyEight na NBA)
ELO_START = 1500.0
ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HOME_ADV = 100.0
ELO_SEASON_CARRY = 0.75  # fração do Elo mantida de uma temporada para a outra

# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
SEED = 42
UNITS = 32
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 6


def season_start_year(season: str) -> int:
    """'2023-24' -> 2023"""
    return int(season[:4])


def season_from_year(year: int) -> str:
    """2023 -> '2023-24'"""
    return f"{year}-{(year + 1) % 100:02d}"


def team_seasons() -> list[str]:
    """Temporadas cujos jogos de time precisam ser baixados
    (temporadas do modelo + HISTORY_SEASONS anteriores)."""
    first = season_start_year(MODEL_SEASONS[0]) - HISTORY_SEASONS
    last = season_start_year(MODEL_SEASONS[-1])
    return [season_from_year(y) for y in range(first, last + 1)]


def player_seasons() -> list[str]:
    """Temporadas cujos logs de jogador precisam ser baixados
    (temporadas do modelo + INJURY_SEASONS anteriores, para o histórico de
    jogos perdidos e a continuidade do elenco)."""
    first = season_start_year(MODEL_SEASONS[0]) - INJURY_SEASONS
    last = season_start_year(MODEL_SEASONS[-1])
    return [season_from_year(y) for y in range(first, last + 1)]


def impact_seasons() -> list[str]:
    """Temporadas cujo rating de impacto é necessário (a anterior a cada temporada do modelo)."""
    return [season_from_year(season_start_year(s) - 1) for s in MODEL_SEASONS]


def hoopr_year(season: str) -> int:
    """O hoopR identifica a temporada pelo ano em que ela TERMINA: '2024-25' -> 2025."""
    return season_start_year(season) + 1


def train_seasons() -> list[str]:
    return [s for s in MODEL_SEASONS if s not in (VAL_SEASON, TEST_SEASON)]
