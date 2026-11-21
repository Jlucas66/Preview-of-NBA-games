# Previsão de jogos da NBA com Redes Neurais Líquidas (CfC) e impacto dos jogadores

Projeto da disciplina de Redes Neurais (UFRPE). O objetivo é prever a **probabilidade de
o mandante vencer** um jogo da NBA a partir de:

- **forma recente** de cada time: sequência dos últimos 10 jogos, com o intervalo real
  em dias entre eles (entrada da rede líquida);
- **impacto de cada jogador**: rating adj-RAPM da temporada anterior (hoopR/SportsDataverse),
  ponderado pelos minutos que ele costuma jogar;
- **montagem do elenco**: impacto de quem chegou e de quem saiu, continuidade;
- **desfalques no dia**: impacto de quem está fora do jogo, se o principal jogador está fora;
- **propensão a lesão**: % de jogos que cada jogador perdeu nas últimas 3 temporadas;
- **contexto**: Elo, mando, descanso, back-to-back, histórico das últimas 5 temporadas.

Modelo principal: **CfC (Closed-form Continuous-time)**, a versão rápida das redes
líquidas LTC. Comparado com LTC, CfC com fiação NCP, LSTM e baselines (sempre o
mandante, Elo, regressão logística).

## Estrutura

```
nba-lnn/
├── src/
│   ├── config.py            # temporadas, métrica de impacto, hiperparâmetros
│   ├── data/
│   │   ├── collect.py       # baixa jogos/jogadores (nba_api) + impacto (hoopR)
│   │   └── synthetic.py     # dados falsos no mesmo formato (teste offline)
│   ├── features/
│   │   ├── build.py         # Elo, descanso, histórico, sequências -> dataset
│   │   └── players.py       # impacto, desfalques, trocas, propensão a lesão
│   ├── models/
│   │   ├── networks.py      # rede de duas torres (CfC / NCP / LTC / LSTM)
│   │   ├── dataset.py       # divisão por temporada + normalização
│   │   ├── baselines.py     # mandante, Elo, regressão logística
│   │   └── metrics.py       # acurácia, log-loss, Brier, AUC
│   ├── train.py             # treino, early stopping, comparação, várias sementes
│   └── analysis.py          # figuras para o artigo
├── app.py                   # site de demonstração (Streamlit)
├── data/raw/                # CSVs brutos (versionados no git p/ reprodutibilidade)
├── data/processed/          # gerado
└── artifacts/               # pesos, previsões, tabelas e figuras (gerado)
```

## Como rodar

```bash
git clone https://github.com/Jlucas66/Preview-of-NBA-games.git
cd Preview-of-NBA-games/nba-lnn             # o projeto fica nesta subpasta

python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python -m src.data.collect                  # 1. baixa os dados (~5-10 min)
python -m src.features.build                # 2. features + sequências
python -m src.train --model all --seeds 5   # 3. treina tudo, 5 sementes por rede
python -m src.analysis                      # 4. figuras em artifacts/figures/
```

Rode sempre a partir da **raiz do projeto**, porque os comandos usam `python -m src...`.

- O passo 1 precisa rodar **no seu computador**. O stats.nba.com costuma bloquear IPs de
  nuvem (Colab, servidores). Depois de baixar, faça commit da pasta `data/raw/`. Assim
  qualquer pessoa, inclusive no Colab, roda os passos 2 a 4 sem acessar a NBA.
- Para testar o pipeline sem internet: `python -m src.data.collect --synthetic`. Os
  resultados com dados sintéticos não valem para a NBA real.
- Para treinar um modelo só: `python -m src.train --model cfc` (ou `ncp`, `ltc`, `lstm`).

## Dados

| Arquivo | Fonte | Conteúdo |
|---|---|---|
| `team_games_<temp>.csv` | `nba_api` → `LeagueGameFinder` | 1 linha por time por jogo (placar, arremessos, rebotes...) |
| `player_games_<temp>.csv` | `nba_api` → `PlayerGameLogs` | 1 linha por jogador por jogo (minutos) |
| `player_impact_<temp>.csv` | [hoopR Player Impact Model](https://hoopr.sportsdataverse.org/articles/nba-player-impact-model.html) (releases do GitHub `sportsdataverse/sportsdataverse-data`) | 1 linha por jogador por temporada: adj-RAPM, RAPM, BPM, SPM, WAR |

O `player_id` do hoopR é o mesmo id do stats.nba.com, então as bases se juntam direto.
O hoopR é um pacote de R, mas os arquivos de impacto são parquet públicos e o projeto
os lê em Python. **Não é preciso instalar R.**

Temporadas (mude em `src/config.py`):
- modelo: 2021-22 a 2025-26 (as últimas 5 temporadas completas). Treino até 2023-24,
  validação 2024-25, teste 2025-26;
- jogos dos times desde 2016-17, para o histórico de 5 temporadas;
- logs de jogadores desde 2018-19, para o histórico de 3 temporadas de jogos perdidos;
- impacto dos jogadores de 2020-21 a 2024-25 (sempre a temporada anterior).

## Features

### Sequência (entrada da CfC)
Para cada time, os 10 jogos anteriores com: vitória, pontos feitos e sofridos, saldo,
FG%, 3P%, FT%, rebotes, assistências, turnovers, estatísticas do adversário, mando,
Elo dos dois times e a força de quem jogou de cada lado.

**Timespans**: os dias entre esses jogos. É aqui que a rede líquida se diferencia. A
constante de tempo dos neurônios depende do intervalo real, então um back-to-back
(1 dia), uma pausa de uma semana e o offseason são tratados de forma diferente.

### Impacto e elenco (`src/features/players.py`)

Cada jogador *p* tem um rating *r_p* (adj-RAPM da temporada anterior; novatos recebem
−1,0) e minutos esperados *m_p* (média dos últimos 10 jogos dele).

| Feature | Fórmula / significado |
|---|---|
| `LINEUP_STRENGTH` | 5 · Σ r_p·m_p / Σ m_p sobre **quem joga hoje** (≈ saldo por 100 posses do quinteto) |
| `ROSTER_STRENGTH` | mesma conta com o elenco inteiro (teto do time saudável) |
| `MISSING_IMPACT` | Σ r_p·m_p/48 de quem está no elenco e **não joga hoje** |
| `MISSING_MIN` | fração dos minutos esperados do elenco que está fora hoje |
| `STAR_OUT` | 1 se o jogador com maior r_p·m_p está fora |
| `INJURY_RISK` | média, ponderada por minutos, da taxa de jogos perdidos de cada jogador nas 3 temporadas anteriores (suavizada para a média da liga) |
| `IMPACT_IN` | impacto dos reforços (não estavam no time na temporada passada) |
| `IMPACT_OUT` | impacto de quem saiu, ponderado pelos minutos que jogava no time |
| `CONTINUITY` | % dos minutos da temporada passada que continuam no elenco |

Assim, se um jogador de rating alto é trocado, `IMPACT_OUT` e `ROSTER_STRENGTH` mudam. Se
ele se machuca, `LINEUP_STRENGTH`, `MISSING_IMPACT` e `STAR_OUT` mudam. Se ele costuma
se machucar, `INJURY_RISK` sobe.

**Elenco** = jogadores que atuaram pelo time nos últimos 15 jogos do time e cujo último
time é este. Isso trata trocas no meio da temporada.

**Jogos perdidos** = o time jogou e o jogador, que estava no elenco, não entrou. O
período no elenco vai do 1º ao último jogo pelo time. Ele é estendido até o fim da
temporada se aquele foi o último time do jogador (pega lesões que encerram a temporada) e
até o início se ele já era do time (pega quem começou lesionado). *Limitação*: os logs
não dizem o motivo da ausência, então lesão, descanso e opção técnica se misturam.

### Outras features
`ELO_PRE` (K=20, +100 de mando, regride 25% entre temporadas), `REST`, `B2B`,
`HIST_WINPCT` (últimas 5 temporadas), `LAST_WINPCT`, `SEASON_PROGRESS`, e as diferenças
`ELO_DIFF` e `STRENGTH_DIFF`.

## Arquitetura

```
últimos 10 jogos do mandante  ─► [CfC] ─► e_m ┐
últimos 10 jogos do visitante ─► [CfC] ─► e_v ┼─► [e_m, e_v, e_m − e_v, estáticas] ─► MLP ─► P(mandante vence)
     (mesmos pesos: rede siamesa)              │
features estáticas (elenco, Elo, ...) ─────────┘
```

Perda: entropia cruzada binária. Otimizador AdamW, early stopping pela log-loss de
validação, *gradient clipping* 1,0.

> O `forward` do pacote `ncps` faz `timespans[:, t].squeeze()`, que quebra quando cada
> exemplo do lote tem um intervalo diferente. Por isso `networks.py` percorre o tempo
> manualmente, chamando a célula (`rnn_cell`) com `ts` no formato `(B, 1)`. Com `ts = 1`
> a saída é idêntica à da biblioteca.

## Evitando vazamento de dados

- Toda feature usa só informação **anterior** ao jogo: o rating e o histórico de lesões
  vêm de temporadas anteriores, e os minutos esperados de jogos anteriores.
- Única suposição: quem entra em quadra é conhecido antes do jogo, pelo injury report
  oficial da NBA.
- A normalização e o preenchimento de ausentes usam **só o treino**.
- A divisão é **por temporada**, sem embaralhar entre temporadas.

Projetos que dizem ter 80% ou mais de acurácia costumam usar estatísticas do próprio
jogo, o que é vazamento. Referências realistas: sempre apostar no mandante ≈ 55–58%;
casas de apostas ≈ 68–70%.

## Métricas e análise

- **Acurácia**, **log-loss**, **Brier** e **AUC** no teste. Para as redes, média ± desvio
  entre sementes.
- `analysis.py` gera: a comparação entre modelos, as curvas de treino, a calibração
  (CfC vs Elo), a importância por permutação de cada grupo de features e o efeito, na
  previsão, de mudar a força de quem joga pelo mandante.

## Site de demonstração (Streamlit)

Depois dos passos 1 a 4:

```bash
python -m src.data.collect --current   # temporada em andamento: agenda, jogos, elencos oficiais
python -m src.features.current         # features dos próximos jogos
streamlit run app.py                   # abre http://localhost:8501
```

Os dois primeiros comandos são opcionais. Sem eles, o site mostra só os jogos disputados.
Rode os dois de novo sempre que quiser atualizar a agenda, os resultados e os elencos.

No site:
- **Próximos jogos (2026-27):** usa o **elenco oficial atual** de cada time
  (`CommonTeamRoster`). A escalação provável é quem jogou em algum dos últimos 3 jogos; os
  outros (lesionados, recém-contratados) começam desmarcados. Corrija pelo injury report.
- **Jogos já disputados:** a temporada de teste (2025-26) ou o que já foi jogado em
  2026-27, com o resultado real. Os jogos de 2026-27 são posteriores a todo o treino: é
  um teste ao vivo do modelo.
- O jogo é escolhido pela **data** (calendário). Os jogos do dia aparecem em lista, e dá
  para filtrar por time.
- **Simulador:** marque ou desmarque quem joga. O app recalcula as features de elenco com
  a mesma função do treino (`players.lineup_features`) e roda a rede de novo.

### Compartilhar com outra pessoa (link temporário do seu PC)

Na mesma rede Wi-Fi, use o endereço *Network URL* que o Streamlit mostra no terminal.

Para alguém em outro lugar, use um túnel gratuito da Cloudflare (não precisa de conta):

```bash
# instalar (Ubuntu/Debian, uma vez)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# com o streamlit rodando em outro terminal:
cloudflared tunnel --url http://localhost:8501
```

O comando mostra um link `https://....trycloudflare.com`. Mande esse link. Ele funciona
enquanto o terminal estiver aberto e o PC ligado, e muda a cada vez que você roda o túnel.

## Repositório no GitHub

O código está em <https://github.com/Jlucas66/Preview-of-NBA-games>, dentro da pasta `nba-lnn/`.

## Rodando no Google Colab

O notebook `notebooks/colab_nba_lnn.ipynb` clona este repositório e roda o pipeline
inteiro: features → treino → figuras.

1. Baixe os dados **no seu computador** (`python -m src.data.collect`) e faça commit de `data/raw/`.
2. O `REPO_URL` do notebook já aponta para este repositório.
3. Abra no Colab: <https://colab.research.google.com/github/Jlucas66/Preview-of-NBA-games/blob/main/nba-lnn/notebooks/colab_nba_lnn.ipynb>.
4. *Ambiente de execução → Executar tudo*. GPU é opcional: em CPU leva poucos minutos.

O repositório precisa ser **público** para o Colab clonar sem senha. Se for privado, use
um token de acesso na URL ou suba o zip do projeto direto no Colab.

## Referências

- Hasani et al. (2021). *Liquid Time-constant Networks*. AAAI.
- Hasani et al. (2022). *Closed-form continuous-time neural networks*. Nature Machine Intelligence.
- Lechner et al. (2020). *Neural circuit policies enabling auditable autonomy*. Nature Machine Intelligence.
- SportsDataverse / hoopR, *NBA Player Impact Model* (RAPM, adj-RAPM, BPM).
- `nba_api`: cliente Python do stats.nba.com.
