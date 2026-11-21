"""Site de demonstração: previsões da rede líquida (CfC) + simulador de desfalques.

Uso (na raiz do projeto):
    streamlit run app.py

Modos:
  * Próximos jogos  -> temporada em andamento, elenco OFICIAL atual e escalação
                       provável (precisa de `collect --current` + `features.current`)
  * Jogos disputados -> temporada de teste (2025-26) ou o que já foi jogado na
                       temporada atual, com o resultado real
Em qualquer jogo o usuário pode tirar/colocar jogadores e ver a previsão mudar.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src import config
from src.features.players import lineup_features
from src.models.dataset import load_splits
from src.models.networks import TwoTowerNet

MODEL_NAMES = {"cfc": "CfC (rede líquida)", "ltc": "LTC (rede líquida)",
               "ncp": "CfC + fiação NCP", "lstm": "LSTM"}
HOME_COLOR, AWAY_COLOR = "#2a6fdb", "#d95f02"
CUR, TEST = config.CURRENT_SEASON, config.TEST_SEASON

st.set_page_config(page_title="NBA · Previsão com Redes Líquidas", page_icon="🏀", layout="wide")


# ---------------------------------------------------------------------------
# Carregamento (em cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_base():
    splits, meta = load_splits()          # só para pegar a normalização do treino
    test = splits["test"]
    models = {}
    for cell in MODEL_NAMES:
        path = config.ARTIFACTS / f"{cell}.pt"
        if path.exists():
            m = TwoTowerNet(seq_dim=test.seq_home.shape[-1], static_dim=test.static.shape[-1],
                            units=config.UNITS, hidden=config.HIDDEN, dropout=config.DROPOUT, cell=cell)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            models[cell] = m.eval()
    return meta, models


@st.cache_resource
def team_names() -> dict[int, tuple[str, str]]:
    """team_id -> (sigla, nome) usando a lista estática do nba_api (offline)."""
    try:
        from nba_api.stats.static import teams
        return {t["id"]: (t["abbreviation"], t["full_name"]) for t in teams.get_teams()}
    except Exception:  # noqa: BLE001
        return {}


@st.cache_resource
def load_source(kind: str):
    """kind = 'test' (temporada de teste) ou 'current' (temporada em andamento)."""
    meta, _ = load_base()
    norm = meta["norm"]
    if kind == "test":
        npz, gcsv, rpq, season = "dataset.npz", "games.csv", "rosters.parquet", TEST
    else:
        npz, gcsv, rpq, season = "current.npz", "current_games.csv", "current_rosters.parquet", CUR
    if not (config.DATA_PROCESSED / npz).exists():
        return None
    d = np.load(config.DATA_PROCESSED / npz)
    games = pd.read_csv(config.DATA_PROCESSED / gcsv, dtype={"GAME_ID": str})
    games["GAME_ID"] = games["GAME_ID"].str.zfill(10)
    if "FUTURE" not in games:
        games["FUTURE"] = 0
    if "GAME_TIME" not in games:
        games["GAME_TIME"] = ""
    keep = (d["season"] == season)
    gids = d["game_id"][keep]
    games = games.set_index("GAME_ID").loc[gids]
    src = {
        "games": games,
        "seq_home": ((d["seq_home"][keep] - norm["seq_mu"]) / norm["seq_sd"]).astype(np.float32),
        "seq_away": ((d["seq_away"][keep] - norm["seq_mu"]) / norm["seq_sd"]).astype(np.float32),
        "ts_home": d["ts_home"][keep], "ts_away": d["ts_away"][keep],
        "pos": {g: i for i, g in enumerate(gids)},
    }
    r = pd.read_parquet(config.DATA_PROCESSED / rpq)
    src["rosters"] = r[r.GAME_ID.isin(set(gids))]
    tgp = config.DATA_RAW / f"team_games_{season}.csv"
    src["pts"], src["abbr"] = {}, {}
    if tgp.exists():
        tg = pd.read_csv(tgp, dtype={"GAME_ID": str})
        tg["GAME_ID"] = tg["GAME_ID"].str.zfill(10)
        src["pts"] = tg.set_index(["GAME_ID", "TEAM_ID"])["PTS"].to_dict()
        src["abbr"] = tg.groupby("TEAM_ID")["TEAM_ABBREVIATION"].last().to_dict()
    sch = config.DATA_RAW / f"schedule_{season}.csv"
    if sch.exists():
        sc = pd.read_csv(sch)
        src["abbr"].update(dict(zip(sc.HOME_TEAM_ID, sc.HOME_ABBR)))
        src["abbr"].update(dict(zip(sc.AWAY_TEAM_ID, sc.AWAY_ABBR)))
    return src


def normalize_static(raw, norm):
    raw = np.where(np.isnan(raw), norm["static_median"], raw)
    return (raw - norm["static_mu"]) / norm["static_sd"]


@torch.no_grad()
def predict(model, src, idx, static_norm):
    """Previsão para os jogos `idx` (array de posições) com estáticas já normalizadas."""
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))  # noqa: E731
    return torch.sigmoid(model(t(src["seq_home"][idx]), t(src["ts_home"][idx]),
                               t(src["seq_away"][idx]), t(src["ts_away"][idx]),
                               t(static_norm))).numpy()


@st.cache_data
def all_predictions(kind: str, cell: str) -> pd.DataFrame:
    meta, models = load_base()
    src = load_source(kind)
    g = src["games"]
    z = normalize_static(g[meta["static_features"]].to_numpy(float), meta["norm"])
    p = predict(models[cell], src, np.arange(len(g)), z)
    return pd.DataFrame({"GAME_ID": g.index, "DATE": g.GAME_DATE.str[:10].to_numpy(),
                         "FUTURE": g.FUTURE.to_numpy(), "y": g.HOME_WIN.to_numpy(), "p": p})


def label_team(team_id, src, full=True):
    abbr = src["abbr"].get(team_id, str(team_id))
    known = team_names().get(team_id)
    if full and known and known[0] == abbr:
        return f"{known[1]} ({abbr})"
    return abbr


def prob_bar(p_home, home, away):
    ph = round(100 * p_home)
    st.markdown(
        f"""<div style="display:flex;height:34px;border-radius:8px;overflow:hidden;
        font-weight:600;color:white;font-size:15px;margin:4px 0 2px 0">
        <div style="width:{ph}%;background:{HOME_COLOR};display:flex;align-items:center;
        padding-left:10px;white-space:nowrap">{home} {ph}%</div>
        <div style="width:{100 - ph}%;background:{AWAY_COLOR};display:flex;align-items:center;
        justify-content:flex-end;padding-right:10px;white-space:nowrap">{100 - ph}% {away}</div></div>""",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------
if not (config.DATA_PROCESSED / "dataset.npz").exists():
    st.error("Dados processados não encontrados. Rode antes:\n\n"
             "`python -m src.features.build` e `python -m src.train --model all`")
    st.stop()
meta, models = load_base()
if not models:
    st.error("Nenhum modelo treinado em artifacts/. Rode `python -m src.train --model all`.")
    st.stop()
static_names, norm = meta["static_features"], meta["norm"]
has_current = load_source("current") is not None

# ---------------------------------------------------------------------------
# Barra lateral: escolha do jogo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏀 Escolha o jogo")
    modes = (["🔮 Próximos jogos"] if has_current else []) + ["📅 Jogos já disputados"]
    mode = st.radio("O que você quer ver?", modes)
    if mode.startswith("📅"):
        options = ([f"{CUR} (até agora)"] if has_current else []) + [f"{TEST} (temporada de teste)"]
        season_opt = st.selectbox("Temporada", options)
        kind = "current" if season_opt.startswith(CUR) else "test"
    else:
        kind = "current"
    src = load_source(kind)
    games = src["games"]
    future_mode = mode.startswith("🔮")
    games = games[games.FUTURE == (1 if future_mode else 0)]

    all_teams = sorted({int(t) for t in pd.concat([games.HOME_TEAM_ID, games.AWAY_TEAM_ID])},
                       key=lambda t: label_team(t, src))
    team = st.selectbox("Filtrar por time (opcional)", [None] + all_teams,
                        format_func=lambda t: "Todos os times" if t is None else label_team(t, src))
    if team is not None:
        games = games[(games.HOME_TEAM_ID == team) | (games.AWAY_TEAM_ID == team)]

    dates = sorted(games.GAME_DATE.str[:10].unique())
    if not dates:
        st.warning("Nenhum jogo encontrado com esse filtro.")
        st.stop()
    dts = [pd.Timestamp(d).date() for d in dates]
    default = dts[0] if future_mode else dts[-1]
    day = st.date_input("Data", value=default, min_value=dts[0], max_value=dts[-1], format="DD/MM/YYYY")
    day_str = day.strftime("%Y-%m-%d")
    if day_str not in dates:
        near = min(dts, key=lambda d: abs((d - day).days))
        st.info(f"Sem jogos em {day:%d/%m}. Mostrando o dia mais próximo: {near:%d/%m}.")
        day_str = near.strftime("%Y-%m-%d")

    day_games = games[games.GAME_DATE.str[:10] == day_str]
    labels = {gid: f"{label_team(int(r.AWAY_TEAM_ID), src, full=False)} @ "
                   f"{label_team(int(r.HOME_TEAM_ID), src, full=False)}"
                   + (f"  ·  {r.GAME_TIME}" if isinstance(r.GAME_TIME, str) and future_mode else "")
              for gid, r in day_games.iterrows()}
    gid = st.radio(f"Jogos em {pd.Timestamp(day_str):%d/%m/%Y} ({len(labels)})",
                   list(labels), format_func=labels.get)
    st.divider()
    cell = st.selectbox("Modelo", list(models), format_func=lambda c: MODEL_NAMES[c])
    st.caption("Mandante em azul, visitante em laranja. "
               "Percentuais = chance de vitória prevista.")

# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.title("Previsão de jogos da NBA com Redes Neurais Líquidas")
tab_prev, tab_perf, tab_about = st.tabs(["Previsão e simulador", "Desempenho dos modelos", "Como funciona"])

row = src["games"].loc[gid]
home_id, away_id = int(row.HOME_TEAM_ID), int(row.AWAY_TEAM_ID)
home, away = label_team(home_id, src, full=False), label_team(away_id, src, full=False)
i = src["pos"][gid]
raw_static = row[static_names].to_numpy(dtype=float)
model = models[cell]
p_model = float(predict(model, src, [i], normalize_static(raw_static, norm)[None])[0])
is_future = bool(row.FUTURE)

with tab_prev:
    st.subheader(f"{label_team(away_id, src)} @ {label_team(home_id, src)}")
    st.caption(f"{pd.Timestamp(row.GAME_DATE):%d/%m/%Y}"
               + (f" · {row.GAME_TIME}" if is_future and isinstance(row.GAME_TIME, str) else "")
               + f" · temporada {row.SEASON}")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{MODEL_NAMES[cell]}: P({home} vence)", f"{p_model:.0%}")
    c2.metric(f"Elo: P({home} vence)", f"{row.ELO_PROB_HOME:.0%}")
    if is_future:
        c3.metric("Resultado", "A disputar")
        c3.caption("Escalação provável: quem jogou em algum dos últimos "
                   f"{config.PROBABLE_LAST_GAMES} jogos. Corrija abaixo com o injury report.")
    else:
        winner = home if row.HOME_WIN == 1 else away
        c3.metric("Resultado real", f"{winner} venceu")
        hp, ap = src["pts"].get((gid, home_id)), src["pts"].get((gid, away_id))
        if hp is not None:
            c3.caption(f"{home} {hp} × {ap} {away}")
    prob_bar(p_model, home, away)
    if not is_future:
        acertou = (p_model >= 0.5) == (row.HOME_WIN == 1)
        st.caption("✅ O modelo acertou o vencedor." if acertou else "❌ O modelo errou o vencedor.")

    st.markdown("### Simulador de escalação")
    st.write("Desmarque **Joga** para tirar um jogador (lesão, descanso) ou marque para "
             "colocar alguém. A previsão é recalculada com a nova escalação."
             + (" O elenco é o **oficial atual** de cada time." if is_future else
                " O elenco é o daquele jogo, na época."))

    def roster_table(team_id):
        r = src["rosters"][(src["rosters"].GAME_ID == gid) & (src["rosters"].TEAM_ID == team_id)].copy()
        r["peso"] = r.RATING * r.EXP_MIN / 48
        r = r.sort_values(["PLAYED", "peso"], ascending=[False, False])
        return pd.DataFrame({
            "Joga": r.PLAYED.to_numpy(),
            "Jogador": r.PLAYER_NAME.to_numpy(),
            "Impacto": r.RATING.round(2).to_numpy(),
            "Min.": r.EXP_MIN.round(1).to_numpy(),
            "% perdidos": (100 * r.INJURY_RATE).round(0).astype(int).to_numpy(),
        }), r

    edited = {}
    cols = st.columns(2)
    for col, team_id, name, color, role in [(cols[0], home_id, home, HOME_COLOR, "mandante"),
                                            (cols[1], away_id, away, AWAY_COLOR, "visitante")]:
        with col:
            st.markdown(f"<h4 style='color:{color};margin-bottom:0'>{label_team(team_id, src)}"
                        f" <span style='font-size:0.7em;color:#888'>({role})</span></h4>",
                        unsafe_allow_html=True)
            table, raw = roster_table(team_id)
            out = st.data_editor(
                table, key=f"ed_{kind}_{gid}_{team_id}", hide_index=True, width="stretch",
                height=36 * (len(table) + 1) + 3,
                disabled=[c for c in table.columns if c != "Joga"],
                column_config={
                    "Joga": st.column_config.CheckboxColumn(width="small"),
                    "Jogador": st.column_config.TextColumn(width="medium"),
                    "Impacto": st.column_config.NumberColumn(
                        format="%+.2f", help="adj-RAPM da temporada anterior: pontos de saldo por "
                                             "100 posses que o jogador adiciona (novatos: −1,0)"),
                    "Min.": st.column_config.NumberColumn(
                        help="Minutos que ele costuma jogar (média dos últimos 10 jogos)"),
                    "% perdidos": st.column_config.NumberColumn(
                        format="%d%%", help="Jogos perdidos nas últimas 3 temporadas (proxy de lesão)"),
                })
            edited[team_id] = (raw, out["Joga"].to_numpy())

    new_static = raw_static.copy()
    changed = False
    for team_id, suf in [(home_id, "_H"), (away_id, "_A")]:
        raw, plays = edited[team_id]
        changed |= bool((plays != raw.PLAYED.to_numpy()).any())
        for k, v in lineup_features(raw.RATING, raw.EXP_MIN, raw.INJURY_RATE, plays).items():
            if k + suf in static_names:
                new_static[static_names.index(k + suf)] = v
    ix = static_names.index
    new_static[ix("STRENGTH_DIFF")] = new_static[ix("LINEUP_STRENGTH_H")] - new_static[ix("LINEUP_STRENGTH_A")]

    if changed:
        p_new = float(predict(model, src, [i], normalize_static(new_static, norm)[None])[0])
        st.markdown("#### Previsão com a sua escalação")
        d1, d2 = st.columns([1, 3])
        d1.metric(f"P({home} vence)", f"{p_new:.0%}", f"{100 * (p_new - p_model):+.1f} p.p.")
        with d2:
            prob_bar(p_new, home, away)
        st.caption(f"Força em quadra (saldo estimado por 100 posses do quinteto): "
                   f"{home} {raw_static[ix('LINEUP_STRENGTH_H')]:+.2f} → {new_static[ix('LINEUP_STRENGTH_H')]:+.2f} · "
                   f"{away} {raw_static[ix('LINEUP_STRENGTH_A')]:+.2f} → {new_static[ix('LINEUP_STRENGTH_A')]:+.2f}")
    else:
        st.info("Altere a coluna **Joga** para simular desfalques ou retornos.")

# ------------------------------ desempenho ------------------------------
with tab_perf:
    res_path = config.ARTIFACTS / "results.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        show = res[["modelo", "accuracy", "log_loss", "brier", "auc", "params"]].copy()
        show.columns = ["Modelo", "Acurácia", "Log-loss", "Brier", "AUC", "Parâmetros"]
        st.subheader(f"Resultados no teste ({TEST})")
        st.dataframe(show.style.format({"Acurácia": "{:.1%}", "Log-loss": "{:.4f}",
                                        "Brier": "{:.4f}", "AUC": "{:.3f}"}),
                     hide_index=True, width="stretch")
        st.caption("Log-loss e Brier: menor é melhor. Acurácia e AUC: maior é melhor. "
                   "mandante = sempre apostar no time da casa.")

    if has_current:
        cp = all_predictions("current", cell)
        played = cp[cp.FUTURE == 0]
        if len(played):
            acc = ((played.p >= 0.5) == (played.y == 1)).mean()
            st.subheader(f"Temporada atual ({CUR}) — jogos já disputados")
            st.metric(f"Acurácia do {MODEL_NAMES[cell]} até agora", f"{acc:.1%}",
                      help="Esses jogos são posteriores a todo o treino: é um teste ao vivo.")
            st.caption(f"{len(played)} jogos")

    mp = all_predictions("test", cell)
    mp["Mês"] = mp.DATE.str[:7]
    mp["Acerto"] = (mp.p >= 0.5) == (mp.y == 1)
    st.subheader(f"Acurácia por mês em {TEST} — {MODEL_NAMES[cell]}")
    st.bar_chart(mp.groupby("Mês")["Acerto"].mean(), y_label="Acurácia")

    fig_dir = config.ARTIFACTS / "figures"
    figs = [("comparacao_modelos", "Comparação dos modelos"), ("importancia", "O que mais pesa na previsão"),
            ("calibracao", "Calibração"), ("efeito_desfalque", "Efeito da força de quem joga")]
    figs = [(f, t) for f, t in figs if (fig_dir / f"{f}.png").exists()]
    if figs:
        cols = st.columns(2)
        for k, (f, t) in enumerate(figs):
            with cols[k % 2]:
                st.markdown(f"**{t}**")
                st.image(str(fig_dir / f"{f}.png"))

# ------------------------------ sobre -----------------------------------
with tab_about:
    doc = Path(__file__).with_name("docs_app.md")
    if doc.exists():
        st.markdown(doc.read_text(encoding="utf-8"))
