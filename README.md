# Flappy Bird com IA aprimorada (Python + Tkinter)

Um clone de Flappy Bird feito em Python usando Tkinter, com dois modos de IA: um classico (Q-Learning) e outro mais avancado (DQN). Voce pode jogar manualmente, assistir a IA jogar ou treinar a IA ao vivo.

O foco do projeto e ser divertido e mostrar, de forma simples, como uma IA pode aprender a jogar. O codigo traz varias melhorias para a IA sem ficar excessivamente tecnico.

---

## O que tem neste projeto

- Jogo completo em janela (Tkinter), com pontuacao e colisoes.
- IA classica (Q-Learning) e IA avancada (DQN).
- Treino rapido sem abrir janela (modo simulacao).
- Treino ao vivo dentro do jogo.
- Ajuste de velocidade do treino.
- Janela extra com estatisticas da IA.
- Modelo pre-treinado incluso (DQN) para testar rapido.

---

## Requisitos

- Python 3.10+ (recomendado)
- Tkinter (normalmente ja vem com o Python no Windows)
- numpy (necessario para alguns modos de IA)
- torch (opcional, melhora o treino da DQN e pode usar GPU se disponivel)

Instalacao rapida de dependencias opcionais:

```bash
pip install numpy
pip install torch
```

> Dica: Se voce estiver no Linux e o Tkinter nao estiver disponivel, instale `python3-tk` pelo gerenciador de pacotes.

---

## Como jogar (modo manual)

```bash
python main.py
```

### Controles

- Espaco ou Seta para cima: pular
- R: reiniciar
- A: ligar/desligar a IA
- L: ligar/desligar aprendizado ao vivo
- + / -: aumentar/diminuir a velocidade do treino
- T: turbo automatico de treino
- S: modo "estavel" (a IA tenta manter a melhor versao durante o treino)

---

## Modos de IA (simples e direto)

### 1) Q-Learning (mais simples)

- Treinar rapido (sem janela):

```bash
python main.py --train
```

- Jogar com a IA:

```bash
python main.py --play-ai
```

- Aprender ao vivo (com janela):

```bash
python main.py --learn
```

> Se nao houver arquivo de treinamento, o jogo comeca com uma tabela vazia e a IA vai aprendendo do zero.

---

### 2) DQN (mais avancada)

- Treinar DQN sem janela (mais rapido):

```bash
python main.py --nn-train
```

- Jogar com a DQN ja treinada:

```bash
python main.py --nn-play
```

- Treinar ao vivo (com janela):

```bash
python main.py --nn-learn
```

#### Opcoes uteis para DQN

- Usar um modelo maior (mais pesado):

```bash
python main.py --nn-train --nn-heavy
```

- Escolher backend (auto / numpy / torch):

```bash
python main.py --nn-train --nn-backend torch
```

> O modo `auto` tenta usar o Torch se estiver instalado. Caso contrario, usa NumPy.

---

## Explicacao simples de como a IA decide

A IA olha principalmente para:

- Distancia ate o proximo cano
- Altura do passaro em relacao ao centro do buraco
- Velocidade de queda/subida

Ela recebe "pontos" quando sobrevive e passa pelos canos e perde pontos quando bate. Com o tempo, ela aprende quais momentos sao bons para pular e quais nao sao.

No modo DQN, a IA usa uma rede neural para generalizar melhor, aprendendo padroes que funcionam mesmo em situacoes novas.

---

## O que esta acontecendo no treino

- Recompensa por sobreviver: ficar vivo ja conta pontos.
- Recompensa por passar canos: grande bonus quando acerta.
- Recompensa por ficar no centro: quanto mais centralizado, melhor.
- Penalidade por excesso de pulos: pular sem necessidade nao e bom.
- Adaptacao de dificuldade: se a IA vai muito bem, o jogo fica mais dificil; se vai mal, fica um pouco mais facil.

Isso faz a IA aprender de forma mais estavel e ficar mais consistente ao longo do tempo.

---

## Arquivos importantes

- `main.py` - todo o jogo e a IA
- `ai_dqn_enhanced.pkl` - modelo DQN pre-treinado
- `ai_qtable_enhanced.pkl` - arquivo gerado quando voce treina o Q-Learning

---

## Dicas rapidas

- Quer so jogar? Use `python main.py`.
- Quer ver a IA jogar logo? Use `python main.py --nn-play`.
- Quer treinar mais rapido? Aumente a velocidade com `+`.
- O treino pode levar um tempo, mas voce ja ve a IA melhorar em poucos minutos.

---

## Por que este projeto e legal

Alem de ser um jogo classico, este projeto mostra na pratica como uma IA pode evoluir com tentativa e erro. E um otimo exemplo para quem quer entender IA aplicada em jogos de forma simples e divertida.

---

## Licenca

MIT License. Veja o arquivo `LICENSE`.
