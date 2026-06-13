# Explicação do Motor RL e Treinamento (Tensor Mage Arena)

Este documento detalha as principais estruturas da sua base de código para o ambiente de Reinforcement Learning (RL), dividindo-se entre a lógica da Engine (Simulação) e a Rede Neural (Treinamento).

## 1. O Ambiente JAX/Numpy (GameEngine)
A classe `GameEngine` atua como o ambiente (environment) simulado onde o agente interage. Foi projetada de forma matricial para ser rápida e headless (sem gargalos de renderização gráfica), ideal para a amostragem em lote.
- **Reward Shaping (`update_rewards`):** Substituímos recompensas esparsas (que só ocorrem no fim do jogo) por um sinal denso. O agente é punido por tempo (forçando-o a ser rápido), punido por campismo e inércia (se a distância dele para onde estava há 20 ticks for curta) e recompensado a cada tick que captura um item dinâmico do chão.
- **Cost Map e Pathfinding Evasivo (A*):** O bot inimigo agora usa uma matriz de custos baseada em floats. Ele penaliza encostar em quinas de paredes (custo 2.0) e projeta as linhas de tiro inimigas injetando custo letal (100.0). Isso transforma a navegação simples numa IA capaz de desviar dinamicamente dos projéteis.
- **Frame Stacking:** O `frame_buffer` empilha visões temporais em um array 3D. Isso resolve o problema de MDP (Markov Decision Process) incompleto, garantindo que o Agente consiga "enxergar" o movimento, e não apenas o frame estático atual.

## 2. A Arquitetura de Rede Neural (ActorCritic)
Implementada em Flax (JAX), é uma rede de múltiplos domínios para processar o estado gerado pela Engine.
- **Fusão de Sensores:** Uma CNN extrai características da matriz visual do mapa. Uma MLP processa dados escalares de saúde e timers de cooldown. Os logits extraídos de ambos são concatenados, unindo as observações.
- **Actor (Policy Heads):** A rede tem múltiplas cabeças contínuas (Vetor de Mira L2-Normalizado) e discretas (Movement, Shoots, Dash, Shield). Todas as cabeças de skill estão **sempre ativas** na rede — o masking é feito pela engine via `skill_mask`.
- **Action Masking Algébrico:** Extremamente importante. O Agente tenta prever logits para o movimento, mas a máscara envia `-1e7` aos logits de rotas ilegais (como atravessar parede sem o buff de noclip ativo). Quando isso passa na Softmax do PPO, a probabilidade cai para literalmente `0.0`.
- **Critic:** Uma cabeça auxiliar de densa para prever a expectativa de retorno (Value) de um estado, usada no cálculo da Vantagem no PPO.

## 3. Currículo e Treinamento Paralelo (Self-Play)
A configuração de treino (`train.py`) está armada para rodar de forma distribuída entre a CPU/GPU em JAX.
- **Adversários Históricos (Pool):** O loop usa aprendizado por currículo, onde periodicamente copia o cérebro (pesos) do modelo e salva num pool. Assim, o Agente sempre joga contra uma amostragem mista de sua "fase antiga".
- **`self_play_forward_pass`:** Joga o agente que está sendo atualizado contra o inimigo (cujos gradientes estão bloqueados via `jax.lax.stop_gradient`, para o JAX não perder tempo calculando derivada da rede congelada).

## 4. Monitoramento e Telemetria (Matplotlib)
Durante o treinamento iterativo do PPO, rastrear apenas os números no terminal é limitante para interpretar a dinâmica de longo prazo do Agente.
- **Acúmulo de Métricas:** O loop agora acumula listas de histórico em três frentes vitais: `history_rewards` (A recompensa média capturada do ambiente a cada epoch), `history_pi_loss` (A perda da política) e `history_v_loss` (A precisão da estimativa do Critic).
- **Auto-Geração de Gráficos:** A cada 50 epochs de treino, a biblioteca `matplotlib` gera automaticamente um mosaico gráfico de 6 eixos na raiz do projeto (`training_metrics.png`). Isso permite que o desenvolvedor observe a evolução em tempo real, sem precisar encerrar o processo.

## 5. Troubleshooting: Overfitting e Mode Collapse
Na prática real de PPO, é comum o Agente sofrer de **Mode Collapse** (A Política - *Pi Loss* - cai para exatamente zero e a pontuação estaciona). Isso ocorre quando a IA descobre um "buraco seguro" nas recompensas (como ficar parado para ignorar penalidades de movimento falho). 
**Como foi solucionado no ambiente atual:**
1. **Multi-Headed Entropy:** O coeficiente de entropia foi elevado de `0.01` para `0.05` e o cálculo de divergência passou a englobar **todas as 7 cabeças de logits discretas** (não só a de movimento). Se o agente viciar em uma única ação repetitiva, a entropia desaba e o JAX insere gradientes agressivos para forçar a exploração.
2. **Recompensa Cíclica (Sum vs Mean):** A métrica do *Curriculum Learning* agora avalia a recompensa **Total da Epoch** (Sum). Com médias positivas, o progresso sobe microscopicamente (`+0.001` por epoch positiva).

## 6. Técnicas de Otimização e Regularização Avançadas
Para escalar o ambiente a níveis de generalização robustos e combater o "esquecimento catastrófico", as seguintes técnicas State-of-the-Art (SOTA) foram injetadas:
- **EMA (Exponential Moving Average):** Mantemos uma "sombra" dos pesos do modelo usando um filtro passa-baixa (`ema_params = ema * 0.99 + params * 0.01`). Isso suaviza saltos ruidosos no Backpropagation e é usado para gerar os oponentes do Self-Play de forma estabilizada.
- **EWC (Elastic Weight Consolidation):** Regularização no gradiente (baseada numa aproximação matriz de Fisher). A cada 0.1 de progresso no currículo, o código "ancora" os tensores atuais em `ewc_anchor`. A Loss Function soma uma penalidade quadrática para proteger os neurônios cruciais que já aprenderam a resolver os estágios anteriores.
- **Gradient Clipping e SGDR:** Gradientes cortados globalmente em `0.5`. O otimizador `optax` agora usa o agendador *Cosine Annealing with Warm Restarts* (SGDR), fazendo o *Learning Rate* resfriar e reaquecer ciclicamente, forçando a rede a escapar de vales mortos.
- **Dense Reward de Mira e Eficiência:** O ambiente penaliza "mira por strafe" (andar lateralmente até alinhar). Para guiar a saída de controle da mira, um *Dense Reward* contínuo é concedido apenas quando o agente dispara com a mira alinhada ao inimigo. Ao final da partida, um Bônus de Precisão (`50 / tiros_disparados`) é somado à recompensa de abate, ensinando o agente a não "farmar" botões e conservar mana.

## 7. Entropia Adaptativa (Lagrangiano) e Estabilidade Numérica
O coeficiente de entropia fixo (`0.05`) foi substituído por um **multiplicador de Lagrange** auto-ajustável (`α = exp(log_α)`). Funciona assim:
- **Definimos um piso mínimo de entropia** (`H_target = 2.0 nats`). Se a entropia da política cair abaixo disso, o gradiente empurra `log_α` para cima, aumentando agressivamente o bônus de exploração. Se a entropia estiver acima, `α` encolhe para não desperdiçar capacidade.
- **Action Masking:** O valor de mascaramento foi reduzido de `-1e9` para `-1e7`. Com `-1e9`, o `exp()` interno do `jax.nn.softmax` underflowa para exatamente `0.0` em float32, e o `log(0)` resultante gera `NaN` silenciosos nos gradientes. Com `-1e7`, a probabilidade da ação mascarada fica em `~1e-30` (efetivamente zero para amostragem), mas o gradiente permanece numericamente definido.
- **EWC Decay:** O `ewc_lambda` agora decai exponencialmente (`λ *= 0.999` a cada epoch) após ser reativado num milestone de progresso.

Para testar modelos de progressos diferentes, use `--progress 0.5` no renderer.

## 8. Sistema de Currículo Estritamente Contínuo (Continuous Curriculum Learning)

### 8.1 Conceito Fundamental
O sistema **NÃO possui fases ou níveis discretos**. Toda a progressão é governada por uma **única variável float**:

```
curriculum_progress: float ∈ [0.0, 1.0]
```

- `0.0` = Início do treino (ambiente ultra-simplificado)
- `1.0` = Dificuldade máxima (combate completo com todas as habilidades)

A transição é **suave e contínua** — não há saltos bruscos de dificuldade.

### 8.2 Os 5 Subsistemas do Currículo

#### A. Interpolação Linear da Física (LERP)
Todos os parâmetros físicos são calculados **a cada tick** usando interpolação linear:

| Parâmetro | Progress = 0.0 | Progress = 1.0 | Fórmula |
|-----------|----------------|----------------|---------|
| **Hitbox do Inimigo** | 12.0 (colossal) | 1.0 (realista) | `lerp(12.0, 1.0, p)` |
| **HP do Inimigo** | 30 | 100 | `lerp(30, 100, p)` |
| **Vel. do Inimigo** | 0.0 (parado) | 0.3 | `lerp(0, 0.3, p)` |
| **Vel. Proj. Inimigo** | 0.5 (vel. player) | 1.5 (3x player) | `lerp(0.5, 1.5, p)` |
| **Intervalo de Tiro** | 40 ticks (lento) | 8 ticks (rápido) | `lerp(40, 8, p)` |
| **Timeout (max ticks)** | 75 | 1000 | `lerp(75, 1000, p)` |
| **Arena Jogável** | 12×12 (central) | 28×28 (total) | `lerp(12, 28, p)` |

#### B. Desbloqueio por Limiar (Threshold Masking)
Habilidades discretas são liberadas quando o `curriculum_progress` ultrapassa marcos específicos:

| Habilidade | Limiar | Justificativa |
|-----------|--------|---------------|
| **Movimento** | `> 0.10` | Fase torreta forçada no início |
| **Ricochete** | `> 0.40` | Projéteis passam a ricochetear |
| **Labirinto** | `> 0.45` | Paredes internas aparecem |
| **A* Pathfinding** | `> 0.80` | Inimigo usa navegação inteligente |
| **Self-Play** | `> 0.95` | Inimigo controlado por rede neural |

**Conceito Chave:** Quando a máscara libera uma skill (ex: Dash em 0.3), a ameaça física calculada pelo LERP já está num nível onde a habilidade começa a ser matematicamente vantajosa, mas ainda não é letalmente obrigatória.

#### C. Gotejamento de Ameaça (Threat Scaling)
A frequência de ataques especiais do inimigo é uma **probabilidade contínua**:

```python
prob_heavy_attack = max(0.0, (progress - 0.5) * 0.1)
```

Isso faz ataques em leque (3 projéteis simultâneos) começarem a aparecer **raramente** em `progress = 0.5` e aumentarem suavemente até `progress = 1.0` (probabilidade de 5% por tick). Esses ataques forçam o uso de Dash e Shield.

#### D. Atualização do Progresso
O loop de treino chama `engine.update_curriculum(+0.001)` a cada epoch onde a recompensa média é positiva:

```python
if epoch_total_reward > 0 and curriculum_progress < 1.0:
    curriculum_progress += 0.001
    env.update_curriculum(0.001)
```

A cada 0.1 de progresso, os pesos são ancorados via EWC e um checkpoint é salvo.

#### E. Arena de Expansão Dinâmica (Jaula de Vidro)
- O **shape da observação NUNCA muda** (30×30×80 + 8×15)
- O grid é fixo em 30×30 desde o início
- Uma **Área Jogável** (Playable Area) cresce com o progress
- Tudo fora da arena é tratado como parede nos canais de observação
- Spawn, movimentação e projéteis são confinados dentro da arena
- Visualmente, a borda da arena é renderizada como uma linha verde

### 8.3 IA Inimiga Unificada
Ao invés de 10+ branches discretos, a IA heurística é governada por probabilidades:

| Comportamento | Condição | Detalhes |
|--------------|----------|----------|
| Congelado | `progress < 0.15` | Sem movimento nem tiro |
| Random Walk | `progress ≥ 0.15` | Move 20% dos ticks, vel. interpolada |
| Tiro | `progress ≥ 0.25` | Frequência escalada pelo intervalo LERP |
| A* Pathfinding | `progress ≥ 0.80` | Perseguição inteligente |
| Ataques Pesados | `progress ≥ 0.50` | Probabilidade crescente |
| Self-Play Neural | `progress ≥ 0.95` | Rede neural congelada como oponente |

## 9. Correções Implementadas no Reward Shaping
1. **Eliminar bônus de mira por tick** → Bônus de mira é dado APENAS no tick em que o agente dispara.
2. **Remover punição retroativa de miss** → Penalidade fixa de `-0.3` por tiro perdido (borda, parede, bounce excessivo).
3. **Custo por Disparo (Anti-Spray)** → Penalidade de `-0.05` por tiro disparado para ensinar conservação de munição.
4. **Timeout Dinâmico via LERP** → De 75 ticks (progress=0.0) a 1000 ticks (progress=1.0).
5. **Reduzir reward de drops** de `+2.0` para `+0.3`.
6. **Aumentar reward de hit** de `+1.0` base para `+3.0`.
7. **Sinal de Abate Dominante** → Recompensa de `+50.0` por kill, penalidade de `-50.0` por timeout.
8. **Penalidades adaptativas** → Penalidade de tempo e campismo escalam com `curriculum_progress`.
9. **Instrumentação Completa** → Contadores por episódio: `shots_fired`, `shots_hit`, `kill_tick`, `episode_result`, e decomposição da recompensa por fonte.
