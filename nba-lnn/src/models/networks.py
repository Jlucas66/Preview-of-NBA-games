"""Arquiteturas.

Todas seguem o mesmo esquema "duas torres" (siamês):

    jogos anteriores do mandante  --> [encoder recorrente] --> e_home
    jogos anteriores do visitante --> [encoder recorrente] --> e_away   (mesmos pesos)
    [e_home, e_away, e_home - e_away, features estáticas] --> MLP --> logit P(mandante vence)

O encoder pode ser:
    cfc   - Closed-form Continuous-time (rede líquida, versão rápida)   <- modelo principal
    ncp   - CfC com fiação esparsa Neural Circuit Policy (AutoNCP)
    ltc   - Liquid Time-Constant original (resolve a EDO numericamente; mais lento)
    lstm  - baseline recorrente clássico (ignora os intervalos de tempo)

As redes líquidas recebem `timespans` = dias entre os jogos. Com isso o estado
interno "decai" de acordo com o tempo real que passou: um intervalo de 1 dia
(back-to-back) é tratado diferente de uma pausa de 7 dias ou do offseason.
"""
import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int, units: int, cell: str):
        super().__init__()
        self.cell = cell
        if cell == "cfc":
            from ncps.torch import CfC
            self.rnn = CfC(input_size, units, batch_first=True, return_sequences=False)
            self.out_dim = units
        elif cell == "ncp":
            from ncps.torch import CfC
            from ncps.wirings import AutoNCP
            out = max(4, units // 4)
            self.rnn = CfC(input_size, AutoNCP(units, out), batch_first=True,
                           return_sequences=False)
            self.out_dim = out
        elif cell == "ltc":
            from ncps.torch import LTC
            self.rnn = LTC(input_size, units, batch_first=True, return_sequences=False)
            self.out_dim = units
        elif cell == "lstm":
            self.rnn = nn.LSTM(input_size, units, batch_first=True)
            self.out_dim = units
        else:
            raise ValueError(f"encoder desconhecido: {cell}")

    def forward(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) jogos anteriores | ts: (B, L) dias desde o jogo anterior."""
        if self.cell == "lstm":
            out, _ = self.rnn(x)
            return out[:, -1]

        # Laço temporal manual sobre a célula líquida. (O forward do ncps faz
        # `timespans[:, t].squeeze()`, que quebra quando cada exemplo do lote tem
        # um intervalo diferente; aqui passamos ts com formato (B, 1).)
        cell = self.rnn.rnn_cell
        h = torch.zeros(x.size(0), self.rnn.state_size, device=x.device)
        for t in range(x.size(1)):
            h_out, h = cell(x[:, t], h, ts[:, t:t + 1])
        # CfC tem uma projeção de saída (fc); LTC já devolve a saída mapeada
        return self.rnn.fc(h_out) if hasattr(self.rnn, "fc") else h_out


class TwoTowerNet(nn.Module):
    def __init__(self, seq_dim: int, static_dim: int, units: int = 32,
                 hidden: int = 64, dropout: float = 0.2, cell: str = "cfc"):
        super().__init__()
        self.encoder = SequenceEncoder(seq_dim, units, cell)
        e = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(3 * e + static_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, xh, tsh, xa, tsa, static):
        eh = self.encoder(xh, tsh)
        ea = self.encoder(xa, tsa)
        z = torch.cat([eh, ea, eh - ea, static], dim=1)
        return self.head(z).squeeze(1)
