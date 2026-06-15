# Guia de Recompensas e Punições (Reward Shaping)

Este documento detalha exatamente como o sistema de **Aprendizado por Reforço** distribui pontos (recompensas) e tira pontos (penalidades) do Agente a cada segundo de jogo. O objetivo da IA é maximizar essa pontuação.

## 🎯 Ações Base do Jogo

Essas são as recompensas principais, ligadas à mecânica de combate:

* **Matar o Inimigo:** `+50.0` (O grande objetivo. O jogo reseta logo após isso.)
* **Acertar um Tiro:** `+1.0` a `+3.0` (Dependendo do multiplicador de combos e distância.)
* **Coletar um Drop (Vida/Bônus):** `+0.3` (Ganho imediato além do efeito do buff.)
* **Penalidade de Tempo:** `-0.01` por frame (Incentiva o bot a terminar a partida o mais rápido possível e não ficar enrolando.)
* **Penalidade de Camp (Acampar):** `-0.02` por frame (Se ficar muito tempo parado no mesmo raio pequeno, o jogo começa a drenar pontos agressivamente.)
* **Levar Dano (Tiros do Inimigo):** `-5.0` por projétil sofrido. (Foi fortemente aumentado para causar mais "medo" e forçar evasão.)
* **Tiros Consecutivos (Combo de Dano):** `-2.0` extras acumulativos por cada tiro consecutivo sofrido dentro da mesma janela de 1 segundo. Ficar parado tankando tiros drenará seus pontos de forma drástica!
* **Errar um Tiro:** `-0.02` (Para evitar que o bot atire loucamente para todos os lados igual uma metralhadora giratória.)
* **Tiro Batendo na Parede:** `-0.05` (Punição leve para ensinar a não desperdiçar balas em paredes.)

---

## ⚙️ Balanceamento

| Parâmetro | Valor | Notas |
|:---|:---|:---|
| HP Base (Player) | 200 | Morre em ~20 tiros (10 dmg cada) |
| HP Base (Inimigo p=1.0) | 200 | Simétrico ao player |
| HP Base (Inimigo p=0.0) | 300 | Saco de pancadas no início |
| HP Base (Treino) | 300 | Simétrico para self-play |

---

## 🧠 FSM Tática — 6 Estados

O sistema tático usa uma **Máquina de Estados Finita (FSM)** com 6 estados claros. Cada estado define o comportamento esperado e distribui recompensas/penalidades em tempo real.

### ⚔️ ATACAR (Estado 0)
| Ação Esperada | Recompensa | Penalidade |
|:---|:---|:---|
| Manter distância ideal (8-15 blocos) | `+0.15` | — |
| Atirar com linha de visão (LOS) | `+0.10` bônus | — |
| Aproximar quando longe (> 15 blocos) | `+0.10` | `-0.10` se não aproximar |
| Recuar quando muito perto (< 8 blocos) | `+0.05` | `-0.05` se não recuar |

**Quando ativa:** HP > 40% E HP inimigo > 20% E distância de combate

---

### 💀 EXECUTAR (Estado 1)
| Ação Esperada | Recompensa | Penalidade |
|:---|:---|:---|
| Fechar distância | `+0.20` | `-0.15` se recuar |
| Atirar sem parar | `+0.15` | `-0.10` se hesitar |
| Atirar de perto (< 5 blocos) | `+0.20` bônus | — |

**Quando ativa:** HP inimigo ≤ 20%

---

### 🛡️ DEFENDER (Estado 2)
| Ação Esperada | Recompensa | Penalidade |
|:---|:---|:---|
| Manter distância > 15 blocos | `+0.10` | — |
| Buscar cobertura (LOS bloqueado) | `+0.10` bônus | — |
| Atirar de longe com LOS (> 12 blocos) | `+0.10` | — |
| Ficar muito perto (< 10 blocos) | — | `-0.15` |

**Quando ativa:** HP agente < HP inimigo E HP agente > 40%

---

### 🏃 FUGIR (Estado 3)
| Ação Esperada | Recompensa | Penalidade |
|:---|:---|:---|
| Maximizar distância rapidamente | `+0.20` | `-0.20` se aproximar |
| Buscar HP drops (< 5 blocos) | `+0.15` bônus | — |
| Fugir devagar | `+0.05` | — |
| Parado quando deveria fugir | — | `-0.10` |
| Atirar durante fuga | — | `-0.05` |

**Quando ativa:** HP agente ≤ 40% E (HP agente < HP inimigo OU projéteis > 3)

---

### 📐 ENCURRALAR (Estado 4)
| Ação Esperada | Recompensa | Penalidade |
|:---|:---|:---|
| Inimigo perto da parede (< 8 blocos) | `+0.15` | — |
| Inimigo no canto (< 4 blocos) | `+0.25` | — |
| Empurrar inimigo para parede (wall_delta < 0) | `+0.10` | `-0.10` se inimigo escapar |
| Atirar enquanto encurrala | `+0.10` | — |
| Se encurralar no processo (< 3 da parede) | — | `-0.10` |

**Quando ativa:** Inimigo a < 8 blocos da parede E HP agente ≥ HP inimigo

---

### 🗺️ CONTROLAR_MAPA (Estado 5)
| Ação Esperada | Recompensa | Penalidade |
|:---|:---|:---|
| Posição central (30% mais central) | `+0.10` | — |
| Posição razoavelmente central (50%) | `+0.05` | — |
| Nas bordas | — | `-0.05` |
| Coletar drops (< 8 blocos) | `+0.10` bônus | — |
| Atirar quando oportuno com LOS | `+0.05` | — |

**Quando ativa:** Estado neutro / default

---

## 🗺️ Mapa BSP

O mapa é gerado usando **BSP (Binary Space Partitioning)**:
- Em `progress < 0.20`: Arena aberta sem paredes internas
- Em `progress 0.20-1.0`: Cômodos aparecem gradualmente (2-6 salas)
- Cômodos são conectados por corredores de 3 blocos de largura
- Áreas de spawn são limpas em um raio de 5 blocos

> *As recompensas táticas são aplicadas a cada frame (30 vezes por segundo) e multiplicadas pelo `intent_scale` (5.0).*
