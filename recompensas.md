# Guia de Recompensas e Punições (Reward Shaping)

Este documento detalha exatamente como o sistema de **Aprendizado por Reforço** distribui pontos (recompensas) e tira pontos (penalidades) do Agente a cada segundo de jogo. O objetivo da IA é maximizar essa pontuação.

## 🎯 Ações Base do Jogo

Essas são as recompensas principais, ligadas à mecânica de combate:

* **Matar o Inimigo:** `+50.0` (O grande objetivo. O jogo reseta logo após isso.)
* **Acertar um Tiro:** `+1.0` a `+3.0` (Dependendo do multiplicador de combos e distância.)
* **Coletar um Drop (Vida/Bônus):** `+0.3` (Ganho imediato além do efeito do buff.)
* **Penalidade de Tempo:** `-0.01` por frame (Incentiva o bot a terminar a partida o mais rápido possível e não ficar enrolando.)
* **Penalidade de Camp (Acampar):** `-0.02` por frame (Se ficar muito tempo parado no mesmo raio pequeno, o jogo começa a drenar pontos agressivamente.)
* **Levar Dano (Tiros do Inimigo):** `-1.0` por projétil sofrido. (Sim, tomar dano é ruim matematicamente!)
* **Tiros Consecutivos (Combo de Dano):** `-0.2` extras acumulativos por cada tiro consecutivo sofrido dentro da mesma janela de 1 segundo. Ficar parado tankando tiros drenará seus pontos de forma exponencial!
* **Errar um Tiro:** `-0.02` (Para evitar que o bot atire loucamente para todos os lados igual uma metralhadora giratória.)
* **Tiro Batendo na Parede:** `-0.05` (Punição leve para ensinar a não desperdiçar balas em paredes.)

---

## 🧠 Intenções Táticas Ativas (Micro-Recompensas)

Quando uma das 11 táticas está **ativada** no cérebro do robô (podendo ser vista no canto da tela no modo de jogo), ele passa a sofrer as seguintes regras *em tempo real* (Os pontos abaixo já estão na escala base final):

### Ofensivas
| Tática | Ação Esperada | Recompensa (Acerto) | Penalidade (Hipocrisia) |
| :--- | :--- | :--- | :--- |
| **Agressivo / Perseguir** | Diminuir a distância até o inimigo. | `+0.05` | `-0.05` |
| **Flanquear** | Movimentar-se transversalmente à visão inimiga. | `+0.05` | `-0.05` |
| **Forçar Canto** | Empurrar inimigo para perto das paredes. | `+0.05` | `-0.05` |
| **All-In Letal** | Atirar quando o inimigo tiver < 20% de vida. | `+0.25` | `-0.10` a `-0.05` |

### Defensivas
| Tática | Ação Esperada | Recompensa (Acerto) | Penalidade (Hipocrisia) |
| :--- | :--- | :--- | :--- |
| **Recuo Evasivo** | Aumentar a distância física do inimigo. | `+0.05` | `-0.05` |
| **Buscar Cobertura** | Ficar com a linha de visão bloqueada (LOS). | `+0.05` | `-0.05` |
| **Movimento Errático** | Mudar bruscamente de direção vetorial. | `+0.025` | `-0.025` |
| **Sobrevivência Tartaruga** | Não atirar e focar apenas em correr/esconder. | `+0.025` | `-0.05` |

### Controle de Mapa
| Tática | Ação Esperada | Recompensa (Acerto) | Penalidade (Hipocrisia) |
| :--- | :--- | :--- | :--- |
| **Territorial** | Permanecer nos 30% mais centrais do mapa. | `+0.025` | `-0.025` |
| **Patrulha (Kiting)** | Manter distância estabilizada longe do inimigo. | `+0.025` | `-0.025` |
| **Caçar Recursos** | Estar fisicamente perto de poções espalhadas. | `+0.10` | `-0.05` |

> *Essas recompensas táticas são concedidas a cada frame (30 vezes por segundo).*
