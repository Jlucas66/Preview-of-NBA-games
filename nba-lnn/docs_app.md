### O que o modelo faz
Prevê a chance de o **mandante vencer** um jogo da NBA, usando só informação disponível **antes** do jogo.

### Entradas
- **Forma recente**: os 10 jogos anteriores de cada time (placar, aproveitamento, Elo, força de quem jogou), junto com os **dias entre os jogos**.
- **Impacto de cada jogador**: o adj-RAPM da temporada anterior (hoopR/SportsDataverse), ou seja, quantos pontos de saldo por 100 posses ele adiciona. O peso dele no jogo é o impacto × os minutos que costuma jogar.
- **Elenco**: quem joga hoje, quem está fora (desfalques), quem chegou e quem saiu desde a temporada passada.
- **Propensão a lesão**: a % de jogos que cada jogador perdeu nas últimas 3 temporadas.
- **Contexto**: Elo, mando de quadra, descanso, back-to-back, histórico das últimas 5 temporadas.

### A rede
Uma **CfC (Closed-form Continuous-time)**, evolução das redes líquidas LTC. A mesma CfC lê a sequência de jogos dos dois times (rede siamesa). Como ela recebe o intervalo real entre os jogos, um back-to-back pesa diferente de uma semana de descanso. A saída da CfC se junta às features de elenco num MLP, que dá a probabilidade de vitória.

### Próximos jogos
Na temporada em andamento, o site usa o **elenco oficial atual** de cada time. A escalação provável é quem jogou em algum dos últimos 3 jogos. Jogadores lesionados há mais tempo ou recém-contratados começam desmarcados: corrija com o injury report do dia.

### Simulador
Ao tirar ou colocar um jogador, o app recalcula, para aquele time, a **força de quem joga**, o **impacto dos desfalques**, se a **estrela está fora** e a **diferença de força** entre os times. Depois roda a rede de novo. A forma recente (os jogos anteriores) não muda.

### Limitações
- A ausência de um jogador pode ser lesão, descanso ou opção técnica: os dados não dizem o motivo.
- Novatos recebem um impacto padrão (−1,0), porque não têm temporada anterior.
- Mesmo as casas de apostas acertam só cerca de 68–70% dos jogos: o basquete tem muita aleatoriedade.
