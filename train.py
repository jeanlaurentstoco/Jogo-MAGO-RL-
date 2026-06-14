import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization
from flax.training import train_state
import optax
import numpy as np
import gc
import time
import copy
import matplotlib.pyplot as plt
from engine import GameEngine
import random

# ==========================================
# CONFIGURAÇÃO DO CURRÍCULO NEURAL (CASCA DE CEBOLA)
# ==========================================
COMPLEXITY_LEVEL = 2
# Nível 1: Engatinhar (1 Frame, sem ResNet) -> MUITO RÁPIDO
# Nível 2: Consolidação (8 Frames, com ResNet) -> BALANCEADO E ROBUSTO
# Skills são controladas pela engine via curriculum_progress (threshold masking)

NUM_FRAMES = 1 if COMPLEXITY_LEVEL == 1 else 8
USE_RESNET = False if COMPLEXITY_LEVEL == 1 else True
CHANNELS_PER_FRAME = 5

class ResNetBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        if residual.shape != x.shape:
            residual = nn.Conv(self.features, kernel_size=(1, 1), padding='SAME')(residual)
        return nn.relu(x + residual)

class EpisodicMemory(nn.Module):
    num_heads: int = 4
    qkv_features: int = 256
    @nn.compact
    def __call__(self, x):
        # x: (Batch, Time, Features)
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, x.shape[1], x.shape[2]))
        x = x + pos_emb
        
        # Transformer Layer 1
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_features)(y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.qkv_features * 2)(y)
        y = nn.relu(y)
        y = nn.Dense(self.qkv_features)(y)
        x = x + y
        
        # Transformer Layer 2
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_features)(y)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.qkv_features * 2)(y)
        y = nn.relu(y)
        y = nn.Dense(self.qkv_features)(y)
        x = x + y
        
        # Extrai o último token temporal (que atendeu a todo o passado) para usar como state context
        return x[:, -1, :]

class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, spatial_obs, scalar_obs, action_mask=None):
        batch_size = spatial_obs.shape[0]
        
        x = spatial_obs.reshape((batch_size, 64, 64, NUM_FRAMES, CHANNELS_PER_FRAME))
        x = jnp.transpose(x, (0, 3, 1, 2, 4)) # (Batch, Time, H, W, C)
        x = x.reshape((batch_size * NUM_FRAMES, 64, 64, CHANNELS_PER_FRAME))
        
        # --- ResNet CNN Backbone ---
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        if USE_RESNET:
            x = ResNetBlock(features=32)(x)
            
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        if USE_RESNET:
            x = ResNetBlock(features=64)(x)
            
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        if USE_RESNET:
            x = ResNetBlock(features=128)(x)
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        x = x.reshape((batch_size, NUM_FRAMES, 128))
        
        y = scalar_obs.reshape((batch_size * NUM_FRAMES, 15))
        y = nn.Dense(64)(y)
        y = nn.relu(y)
        y = y.reshape((batch_size, NUM_FRAMES, 64))
        
        z = jnp.concatenate([x, y], axis=-1)
        z = nn.Dense(128)(z)
        z = nn.relu(z) # (Batch, Time, 128)
        
        # --- Memória Episódica (Self-Attention / Transformer) ---
        mem = EpisodicMemory(num_heads=4, qkv_features=128)(z) # (Batch, 128)
        
        h = nn.Dense(512)(mem)
        h = nn.relu(h)
        h = nn.Dense(512)(h)
        h = nn.relu(h)
        h = nn.Dense(256)(h)
        h = nn.relu(h)
        
        move_logits = nn.Dense(5)(h)
        if action_mask is not None:
            move_logits = jnp.where(action_mask, move_logits, -1e7)
            
        shoot_logits = nn.Dense(2)(h)
        
        aim_vec = nn.Dense(2)(h)
        aim_norm = aim_vec / (jnp.linalg.norm(aim_vec, axis=-1, keepdims=True) + 1e-8)
        
        value = nn.Dense(1)(h)
        
        logits_dict = {
            "move": move_logits,
            "shoot": shoot_logits,
            "aim": aim_norm
        }
        
        return logits_dict, value

# =============================================
# DICIONÁRIO DE INTENÇÕES TÁTICAS (Heurístico para Treinamento)
# =============================================
INTENT_NAMES = [
    "AGRESSIVO_PERSEGUIR",    # 0
    "FLANQUEAR",              # 1
    "FORCAR_CANTO",           # 2
    "ALL_IN_LETAL",           # 3
    "EVASIVO_RECUAR",         # 4
    "BUSCAR_COBERTURA",       # 5
    "MOVIMENTO_ERRATICO",     # 6
    "SOBREVIVENCIA_TARTARUGA",# 7
    "CONTROLE_TERRITORIAL",   # 8
    "PATRULHA_PERIMETRAL",    # 9
    "CAZAR_RECURSOS",         # 10
]

def heuristic_intent(obs, i):
    """Seleciona a intenção tática via regras heurísticas.
    
    Args:
        obs: Dict de observações do VectorEnv
        i: Índice do ambiente
    Returns:
        Vetor one-hot de 11 dimensões
    """
    intent = np.zeros(11, dtype=np.float32)
    
    p_hp = obs["player_stats"][i, 0]
    e_hp = obs["enemy_stats"][i, 0]
    dist = np.linalg.norm(obs["player_pos"][i] - obs["enemy_pos"][i])
    
    # Lógica de prioridade: situações mais urgentes primeiro
    if e_hp <= 10:                              # Inimigo a um tiro de morrer
        intent[3] = 1.0   # ALL_IN_LETAL
    elif p_hp < 30:                              # HP crítico
        intent[4] = 1.0   # EVASIVO_RECUAR
    elif e_hp < 50 and dist > 15:                # Inimigo ferido + longe
        intent[0] = 1.0   # AGRESSIVO_PERSEGUIR
    elif dist < 8:                               # Muito perto
        intent[6] = 1.0   # MOVIMENTO_ERRATICO
    elif dist > 30:                              # Muito longe
        intent[0] = 1.0   # AGRESSIVO_PERSEGUIR
    else:                                        # Neutro
        intent[8] = 1.0   # CONTROLE_TERRITORIAL
    
    # Overlay: se existem drops próximos e HP nao eh crítico
    if p_hp > 50:
        drop_positions = np.argwhere(obs.get("drops", np.zeros((64, 64))) == 1) if "drops" in obs else []
        if len(drop_positions) > 0:
            player_pos = obs["player_pos"][i]
            dists_to_drops = np.linalg.norm(drop_positions - player_pos, axis=1)
            if np.min(dists_to_drops) < 8.0:
                intent = np.zeros(11, dtype=np.float32)
                intent[10] = 1.0  # CAZAR_RECURSOS
    
    return intent

def _worker(remote, parent_remote, num_frames):
    parent_remote.close()
    env = GameEngine(headless=True, num_frames=num_frames, is_training=True)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action, enemy_action, curriculum_progress, hp = data
                env.player_hp_base = hp
                env.enemy_hp_min = hp
                env.enemy_hp_max = hp
                obs, reward, done, _ = env.step(action, enemy_action=enemy_action, curriculum_progress=curriculum_progress)
                ep_stats = None
                if done:
                    ep_stats = env.get_episode_stats()
                    obs = env.reset(curriculum_progress=curriculum_progress)
                remote.send((obs, reward, done, ep_stats))
            elif cmd == 'reset':
                curriculum_progress, hp = data
                env.player_hp_base = hp
                env.enemy_hp_min = hp
                env.enemy_hp_max = hp
                obs = env.reset(curriculum_progress=curriculum_progress)
                remote.send(obs)
            elif cmd == 'get_state':
                remote.send(env.get_state())
            elif cmd == 'update_curriculum':
                increment = data
                env.update_curriculum(increment)
                remote.send(True)
            elif cmd == 'set_intent':
                env.current_intent = np.array(data, dtype=np.float32)
                remote.send(True)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Comando {cmd} não implementado.")
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass

class VectorEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [
            mp.Process(target=_worker, args=(work_remote, remote, NUM_FRAMES))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
            
    def step_async(self, actions_list, enemy_actions_list=None, curriculum_progress=0.0, hp=100.0):
        for i, remote in enumerate(self.remotes):
            ea = enemy_actions_list[i] if enemy_actions_list else None
            remote.send(('step', (actions_list[i], ea, curriculum_progress, hp)))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_list, r_list, done_list, ep_stats_list = zip(*results)
        return self._stack_obs(obs_list), np.array(r_list), np.array(done_list), list(ep_stats_list)

    def step(self, actions_list, enemy_actions_list=None, curriculum_progress=0.0, hp=100.0):
        self.step_async(actions_list, enemy_actions_list, curriculum_progress, hp)
        return self.step_wait()

    def reset(self, curriculum_progress=0.0, hp=100.0):
        for remote in self.remotes:
            remote.send(('reset', (curriculum_progress, hp)))
        obs_list = [remote.recv() for remote in self.remotes]
        return self._stack_obs(obs_list)

    def update_curriculum(self, increment):
        """Envia incremento de curriculum para todos os workers."""
        for remote in self.remotes:
            remote.send(('update_curriculum', increment))
        for remote in self.remotes:
            remote.recv()
    
    def set_intent(self, intents_list):
        """Envia vetor de intenção tática para cada worker.
        
        Args:
            intents_list: Lista de vetores one-hot (num_envs, 11)
        """
        for i, remote in enumerate(self.remotes):
            remote.send(('set_intent', intents_list[i]))
        for remote in self.remotes:
            remote.recv()

    def get_state(self, index=0):
        self.remotes[index].send(('get_state', None))
        return self.remotes[index].recv()

    def close(self):
        if self.closed:
            return
        try:
            if self.waiting:
                for remote in self.remotes:
                    if remote.poll():
                        remote.recv()
            for remote in self.remotes:
                remote.send(('close', None))
        except (EOFError, OSError):
            pass
        finally:
            for p in self.ps:
                p.join(timeout=1.0)
            self.closed = True

    def _stack_obs(self, obs_list):
        return {
            "spatial_obs": np.stack([o["spatial_obs"] for o in obs_list]),
            "scalar_obs": np.stack([o["scalar_obs"] for o in obs_list]),
            "action_mask": np.stack([o["action_mask"] for o in obs_list]),
            "skill_mask": [o["skill_mask"] for o in obs_list],
            "player_pos": np.stack([o["player_pos"] for o in obs_list]),
            "enemy_pos": np.stack([o["enemy_pos"] for o in obs_list]),
            "player_stats": np.stack([o["player_stats"] for o in obs_list]),
            "enemy_stats": np.stack([o["enemy_stats"] for o in obs_list]),
            "curriculum_progress": obs_list[0].get("curriculum_progress", 0.0),
        }

def create_train_state(rng, model, loaded_params=None):
    dummy_spatial = jnp.zeros((1, 64, 64, NUM_FRAMES * CHANNELS_PER_FRAME))
    dummy_scalar = jnp.zeros((1, NUM_FRAMES, 15))
    dummy_mask = jnp.ones((1, 5), dtype=bool)
    if loaded_params is not None:
        params = loaded_params
    else:
        params = model.init(rng, dummy_spatial, dummy_scalar, dummy_mask)
    
    lr_schedule = optax.sgdr_schedule([
        {"init_value": 0.0, "peak_value": 3e-4, "warmup_steps": 50, "decay_steps": 500 * (2 ** i)} for i in range(12)
    ])
    
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr_schedule, eps=1e-5)
    )
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit(static_argnames=['is_training'])
def act_and_explore(params, spatial, scalar, mask, key, is_training=True):
    model = ActorCritic()
    logits, value = model.apply(params, spatial, scalar, mask)
    
    keys = jax.random.split(key, 3)
    
    def sample_categorical(l, k):
        action = jax.lax.cond(
            is_training,
            lambda _: jax.random.categorical(k, l),
            lambda _: jnp.argmax(l, axis=-1),
            operand=None
        )
        log_prob = jax.nn.log_softmax(l)[jnp.arange(l.shape[0]), action]
        return action, log_prob

    actions = {}
    log_probs = {}
    
    actions["move_idx"], log_probs["move"] = sample_categorical(logits["move"], keys[0])
    actions["shoot"], log_probs["shoot"] = sample_categorical(logits["shoot"], keys[1])
    
    actions["aim"] = logits["aim"]
    
    total_log_prob = log_probs["move"] + log_probs["shoot"]
    
    return actions, value[:, 0], total_log_prob


def safe_masked_entropy(logits, mask):
    safe_logits = jnp.where(mask, logits, -1e7)
    log_p = jax.nn.log_softmax(safe_logits, axis=-1)
    p = jax.nn.softmax(safe_logits, axis=-1)
    p_log_p = jnp.where(mask, p * log_p, 0.0)
    return -jnp.sum(p_log_p, axis=-1)

@jax.jit
def train_step(state, batch_spatial, batch_scalar, batch_mask, batch_actions, batch_old_logprobs, batch_returns, batch_advantages, ewc_lambda, ewc_anchor, ewc_fisher):
    
    ENT_COEF = 0.01
    
    def loss_fn(params):
        logits_dict, values = state.apply_fn(params, batch_spatial, batch_scalar, batch_mask)
        values = jnp.squeeze(values)
        
        def get_log_prob(logits, action_idx):
            return jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), action_idx]
            
        lp_move = get_log_prob(logits_dict["move"], batch_actions[:, 0])
        lp_shoot = get_log_prob(logits_dict["shoot"], batch_actions[:, 1])
        
        new_logprobs = lp_move + lp_shoot
        
        ratio = jnp.exp(new_logprobs - batch_old_logprobs)
        clip_adv = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * batch_advantages
        policy_loss = -jnp.mean(jnp.minimum(ratio * batch_advantages, clip_adv))
        
        value_loss = 0.5 * jnp.mean((batch_returns - values) ** 2)
        
        entropy = 0.0
        for key in ["move", "shoot"]:
            if key == "move":
                ent = safe_masked_entropy(logits_dict[key], batch_mask)
                entropy += jnp.mean(ent)
            else:
                logp = jax.nn.log_softmax(logits_dict[key], axis=-1)
                p = jax.nn.softmax(logits_dict[key], axis=-1)
                entropy += -jnp.mean(jnp.sum(p * logp, axis=-1))
                
        # Auxiliary loss para mira supervisionada (Behavior Cloning)
        # scalar_obs tem shape (Batch, Time, 15). Os últimos 2 elementos são o ideal_aim (13:15)
        batch_optimal_aim = batch_scalar[:, -1, 13:15]
        aim_loss = jnp.mean((logits_dict["aim"] - batch_optimal_aim) ** 2)
            
        ewc_loss_leaves = jax.tree_util.tree_map(
            lambda p, a, f: jnp.sum(f * (p - a)**2),
            params, ewc_anchor, ewc_fisher
        )
        ewc_loss = jnp.sum(jnp.array(jax.tree_util.tree_leaves(ewc_loss_leaves)))
        total_loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy + ewc_lambda * ewc_loss + 5.0 * aim_loss
        return total_loss, (policy_loss, value_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (pi_loss, v_loss, ent)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Atualiza Média Móvel da Matriz Fisher (EMA) com o quadrado dos gradientes (Aproximação Diagonal)
    new_fisher = jax.tree_util.tree_map(lambda f, g: 0.99 * f + 0.01 * (g ** 2), ewc_fisher, grads)
    
    return state, loss, pi_loss, v_loss, ent, new_fisher

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns

def load_and_grow_checkpoint(filepath, model, rng, dummy_spatial, dummy_scalar, dummy_mask):
    print(f"=====================================================")
    print(f"Cirurgião de Pesos: Analisando {filepath}...")
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    old_state = flax.serialization.msgpack_restore(bytes_data)
    
    new_state = model.init(rng, dummy_spatial, dummy_scalar, dummy_mask)
    new_params = flax.core.unfreeze(new_state["params"])
    old_params = old_state["params"] if "params" in old_state else old_state
    
    def recursive_update(target, source, path=""):
        for k, v in target.items():
            if isinstance(v, dict):
                if k in source:
                    recursive_update(target[k], source[k], path + f"/{k}")
            else:
                if k in source:
                    old_v = source[k]
                    # Numpy convert to handle JAX arrays
                    v_np = np.array(v)
                    old_v_np = np.array(old_v)
                    
                    if v_np.shape == old_v_np.shape:
                        target[k] = jnp.array(old_v_np)
                    else:
                        print(f"  [>] Adaptando shape em {path}/{k}: de {old_v_np.shape} para {v_np.shape}")
                        if len(v_np.shape) == 4 and "Conv_0" in path:
                            new_v = np.zeros(v_np.shape, dtype=np.float32)
                            c_in_diff = v_np.shape[2] - old_v_np.shape[2]
                            new_v[:, :, c_in_diff:, :] = old_v_np
                            target[k] = jnp.array(new_v)
                        elif len(v_np.shape) == 3 and "pos_emb" in path:
                            new_v = np.zeros(v_np.shape, dtype=np.float32)
                            if old_v_np.shape[1] == 1:
                                for t in range(v_np.shape[1]):
                                    new_v[:, t:t+1, :] = old_v_np
                            target[k] = jnp.array(new_v)
                        else:
                            print(f"  [Aviso] Shape mismatch não tratado em {path}/{k}")
                            target[k] = jnp.array(old_v_np)
                            
    recursive_update(new_params, old_params)
    print(f"Cirurgia concluída com sucesso!")
    print(f"=====================================================")
    return flax.core.freeze({"params": new_params})

def run_training_loop(epochs=100000):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    print("Dispositivos JAX detectados:", jax.devices())
    rng = jax.random.PRNGKey(42)
    model = ActorCritic()
    
    dummy_spatial = jnp.zeros((1, 64, 64, NUM_FRAMES * CHANNELS_PER_FRAME))
    dummy_scalar = jnp.zeros((1, NUM_FRAMES, 15))
    dummy_mask = jnp.ones((1, 5), dtype=bool)
    
    import os
    import glob
    
    checkpoints = glob.glob("modelo_progress_*.msgpack")
    latest_progress = -1.0
    checkpoint_file = None
    
    for ckpt in checkpoints:
        try:
            prog = float(ckpt.split('_')[-1].replace('.msgpack', ''))
            if prog > latest_progress:
                latest_progress = prog
                checkpoint_file = ckpt
        except ValueError:
            pass
            
    if os.path.exists("modelo_treinado.msgpack") and checkpoint_file is None:
        checkpoint_file = "modelo_treinado.msgpack"
        latest_progress = 0.0

    if checkpoint_file:
        curriculum_progress = max(0.0, latest_progress) if latest_progress >= 0 else 0.0
        print(f"Carregando pesos de: {checkpoint_file} (Progress detectado: {curriculum_progress})")
        params = load_and_grow_checkpoint(checkpoint_file, model, rng, dummy_spatial, dummy_scalar, dummy_mask)
        state = create_train_state(rng, model, loaded_params=params)
    else:
        curriculum_progress = 0.0
        state = create_train_state(rng, model)
    
    num_envs = 8
    steps_per_epoch = 128 # 8 * 128 = 1024 passos totais por epoch (reduzido para caber na VRAM)
    env = VectorEnv(num_envs=num_envs)
    
    print("=====================================================")
    print(f"Iniciando Treinamento PPO Robusto com ResNet + Memória Episódica (Self-Attention)")
    print(f"Total Envs: {num_envs} | Steps/Epoch: {steps_per_epoch*num_envs}")
    print("DICA: Pressione Ctrl+C para interromper o treinamento a qualquer momento!")
    print("=====================================================\n")
    
    # curriculum_progress inicializado no bloco de carregamento acima
    historical_params_pool = []
    
    obs = env.reset(curriculum_progress=curriculum_progress)
    
    ewc_anchor = state.params
    ewc_fisher = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.params)
    ewc_lambda = 0.0
    ewc_reset_selfplay = False
    
    opp_params = state.params
    
    history_rewards = []
    history_sr = []
    history_pi_loss = []
    history_v_loss = []
    history_accuracy = []
    history_entropy = []
    history_v_mean = []
    history_progress = []
    live_renderer = None
    last_opp_update_epoch = 0
    
    try:
        for epoch in range(1, epochs + 1):
            start_t = time.time()
            
            # --- INTERVENÇÕES DE SELF-PLAY ---
            chance_self_play = max(0.0, min(1.0, (curriculum_progress - 0.80) / 0.20))
            if chance_self_play > 0.5 and not ewc_reset_selfplay:
                ewc_lambda = 0.05
                ewc_fisher = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.params)
                ewc_anchor = state.params
                ewc_reset_selfplay = True
                print("\n>>> [EWC] Limiar de Self-Play crítico atingido! EWC reiniciado (Fisher zerado) para adaptação livre.\n")
            
            supervisionar = os.path.exists("ver.txt")
            if supervisionar and live_renderer is None:
                from renderer import Renderer
                live_renderer = Renderer()
            elif not supervisionar and live_renderer is not None:
                import pygame
                pygame.quit()
                live_renderer = None
            
            spatial_list, scalar_list, mask_list = [], [], []
            actions_list, logprob_list, reward_list, value_list, done_list = [], [], [], [], []
            epoch_episode_stats = []  # Acumula stats de episódios finalizados nesta epoch
            
            for _ in range(steps_per_epoch):
                # --- INTENÇÃO TÁTICA HEURÍSTICA ---
                intents = [heuristic_intent(obs, i) for i in range(num_envs)]
                env.set_intent(intents)
                
                spatial = jnp.array(obs["spatial_obs"])
                scalar = jnp.array(obs["scalar_obs"])
                mask = jnp.array(obs["action_mask"])
                
                # Dynamic HP for Self-Play
                current_hp = 100.0
                if chance_self_play > 0.0:
                    current_hp = max(100.0, 300.0 - (epoch // 10))
                
                rng, subkey = jax.random.split(rng)
                actions_dict, val, log_prob = act_and_explore(state.params, spatial, scalar, mask, subkey, is_training=True)
                
                env_actions = []
                for i in range(num_envs):
                    act = {
                        "move_idx": int(actions_dict["move_idx"][i]),
                        "shoot": bool(actions_dict["shoot"][i]),
                        "aim": np.array(actions_dict["aim"][i])
                    }
                    env_actions.append(act)
                
                enemy_actions = None
                if np.random.rand() < chance_self_play:  # Soft Self-Play progressivo
                    dice = np.random.rand()
                    use_ia = False
                    curr_opp_params = opp_params
                    
                    if dice < 0.5:
                        pass # 50% de chance: IA heuristica das fases antigas (enemy_actions = None)
                    elif dice < 0.75:
                        # 25% de chance: Oponente do passado
                        curr_opp_params = random.choice(historical_params_pool) if len(historical_params_pool) > 0 else opp_params
                        use_ia = True
                    else:
                        # 25% de chance: Oponente mais recente
                        use_ia = True
                        
                    if use_ia:
                        enemy_actions = []
                        e_spatial = obs["spatial_obs"].copy()
                    
                        # Inverte perspectiva nos ultimos frames (indices 2 e 3 de cada bloco de 5)
                        # spatial_obs é (Batch, 64, 64, 80)
                        for frame_idx in range(NUM_FRAMES):
                            start_idx = frame_idx * CHANNELS_PER_FRAME
                            temp = e_spatial[:, :, :, start_idx + 2].copy()
                            e_spatial[:, :, :, start_idx + 2] = e_spatial[:, :, :, start_idx + 3]
                            e_spatial[:, :, :, start_idx + 3] = temp
                            
                        # Constrói os atributos escalares na perspectiva do inimigo (tamanho 15)
                        # Modelo espera: 10 (agente) + 3 (oponente) + 2 (aim PINN)
                        pad_enemy = np.zeros((num_envs, 10), dtype=np.float32)
                        pad_enemy[:, :3] = obs["enemy_stats"][:, :3]
                        
                        e_opp_stats = obs["player_stats"][:, :3]
                        
                        to_player = obs["player_pos"] - obs["enemy_pos"]
                        norms_e = np.linalg.norm(to_player, axis=1, keepdims=True)
                        norms_safe = np.where(norms_e > 0, norms_e, 1.0)
                        enemy_aim = np.where(norms_e > 0, to_player / norms_safe, np.array([[0.0, 1.0]], dtype=np.float32))
                        
                        e_scalar = np.concatenate([pad_enemy, e_opp_stats, enemy_aim], axis=1).astype(np.float32)
                        # Broadcast no tempo
                        e_scalar = np.repeat(e_scalar[:, None, :], NUM_FRAMES, axis=1) 
                        e_mask = np.ones((num_envs, 5), dtype=bool)
                        
                        rng, subkey2 = jax.random.split(rng)
                        
                        # --- INTERVENÇÃO 3: Ruído Térmico (Epsilon-Greedy no Self-Play) ---
                        if np.random.rand() < 0.05:
                            e_act_dict, _, _ = act_and_explore(curr_opp_params, jnp.array(e_spatial), jnp.array(e_scalar), jnp.array(e_mask), subkey2, is_training=True)
                        else:
                            e_act_dict, _, _ = act_and_explore(curr_opp_params, jnp.array(e_spatial), jnp.array(e_scalar), jnp.array(e_mask), subkey2, is_training=False)
                            
                        for i in range(num_envs):
                            e_act = {
                                "move_idx": int(e_act_dict["move_idx"][i]),
                                "shoot": bool(e_act_dict["shoot"][i]),
                                "aim": np.array(e_act_dict["aim"][i])
                            }
                            enemy_actions.append(e_act)
                
                next_obs, reward, done, ep_stats_list = env.step(env_actions, enemy_actions_list=enemy_actions, curriculum_progress=curriculum_progress, hp=current_hp)
                
                # Coleta stats de episódios finalizados
                for ep_stat in ep_stats_list:
                    if ep_stat is not None:
                        epoch_episode_stats.append(ep_stat)
                
                if supervisionar and live_renderer is not None:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            if os.path.exists("ver.txt"): os.remove("ver.txt")
                            supervisionar = False
                            pygame.quit()
                            live_renderer = None
                    if supervisionar and live_renderer is not None:
                        state_render = env.get_state(0)
                        live_renderer.render(state_render)
                        time.sleep(0.02)
                
                spatial_list.append(obs["spatial_obs"])
                scalar_list.append(obs["scalar_obs"])
                mask_list.append(obs["action_mask"])
                
                act_arr = np.stack([
                    actions_dict["move_idx"], actions_dict["shoot"]
                ], axis=1)
                
                actions_list.append(act_arr)
                logprob_list.append(np.array(log_prob))
                
                # Reward Clipping / Normalization (Aumentado para acomodar grandes recompensas de abate)
                clipped_reward = np.clip(reward, -100.0, 100.0)
                reward_list.append(np.array(clipped_reward))
                
                value_list.append(np.array(val))
                done_list.append(np.array(done))
                
                obs = next_obs
                    
            spatial = jnp.array(obs["spatial_obs"])
            scalar = jnp.array(obs["scalar_obs"])
            mask = jnp.array(obs["action_mask"])
            _, next_val, _ = act_and_explore(state.params, spatial, scalar, mask, rng, is_training=True)
            
            advantages, returns = compute_gae(
                np.array(reward_list), np.array(value_list), 
                np.array(done_list), np.array(next_val)
            )
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            b_spatial = jnp.array(spatial_list).reshape((-1, 64, 64, NUM_FRAMES * CHANNELS_PER_FRAME))
            b_scalar = jnp.array(scalar_list).reshape((-1, NUM_FRAMES, 15))
            b_mask = jnp.array(mask_list).reshape((-1, 5))
            b_actions = jnp.array(actions_list, dtype=jnp.int32).reshape((-1, 2))
            b_logprobs = jnp.array(logprob_list).reshape((-1,))
            b_returns = jnp.array(returns).reshape((-1,))
            b_advantages = jnp.array(advantages).reshape((-1,))
            
            batch_size = b_spatial.shape[0]
            indices = np.arange(batch_size)
            np.random.shuffle(indices)
            
            mini_batch_size = 64
            ppo_epochs = 4
            total_pi_loss, total_v_loss = 0.0, 0.0
            steps_count = 0
            
            for inner_epoch in range(ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    mb_idx = indices[start:end]
                    
                    state, loss, pi_loss, v_loss, ent, ewc_fisher = train_step(
                        state, b_spatial[mb_idx], b_scalar[mb_idx], b_mask[mb_idx], 
                        b_actions[mb_idx], b_logprobs[mb_idx], 
                        b_returns[mb_idx], b_advantages[mb_idx],
                        ewc_lambda, ewc_anchor, ewc_fisher
                    )
                    
                    loss_val = float(loss)
                    if np.isnan(loss_val) or np.isinf(loss_val):
                        print(f"\nCRÍTICO: Colapso numérico detectado na Epoch {epoch}. Interrompendo treinamento.")
                        print("O último checkpoint salvo em disco está intacto e livre da corrupção.")
                        env.close()
                        return
                    
                    total_pi_loss += pi_loss
                    total_v_loss += v_loss
                    steps_count += 1
                
            pi_loss = float(total_pi_loss) / steps_count
            v_loss = float(total_v_loss) / steps_count
            
            # Libera buffers de rollout da GPU para evitar OOM
            del b_spatial, b_scalar, b_mask, b_actions, b_logprobs, b_returns, b_advantages
            gc.collect()
            
            
            
            ewc_lambda = max(0.0, ewc_lambda * 0.999)
            
            epoch_total_reward = float(np.mean(np.sum(reward_list, axis=0)))
            history_rewards.append(epoch_total_reward)
            history_pi_loss.append(float(pi_loss))
            history_v_loss.append(float(v_loss))
            history_entropy.append(float(ent))
            history_progress.append(curriculum_progress)
            
            # Valor médio previsto pelo Critic nesta epoch
            v_mean_epoch = float(np.mean(value_list))
            history_v_mean.append(v_mean_epoch)
            
            # Vantagem média
            adv_mean_epoch = float(np.mean(advantages))
            
            fps = (steps_per_epoch * num_envs) / (time.time() - start_t)
            
            # --- Calcular Métricas de Episódio a partir dos stats coletados ---
            n_episodes = len(epoch_episode_stats)
            if n_episodes > 0:
                kills = sum(1 for s in epoch_episode_stats if s['episode_result'] == 'kill')
                timeouts = sum(1 for s in epoch_episode_stats if s['episode_result'] == 'timeout')
                deaths = sum(1 for s in epoch_episode_stats if s['episode_result'] == 'death')
                
                sr = (kills / n_episodes) * 100.0
                timeout_rate = (timeouts / n_episodes) * 100.0
                
                total_shots = sum(s['shots_fired'] for s in epoch_episode_stats)
                total_hits = sum(s['shots_hit'] for s in epoch_episode_stats)
                accuracy = (total_hits / max(1, total_shots)) * 100.0
                avg_shots = total_shots / n_episodes
                
                kill_ticks = [s['kill_tick'] for s in epoch_episode_stats if s['kill_tick'] >= 0]
                avg_kill_tick = np.mean(kill_ticks) if kill_ticks else -1
                
                # Decomposição de recompensa média por episódio
                r_kills_avg = np.mean([s['reward_from_kills'] for s in epoch_episode_stats])
                r_hits_avg = np.mean([s['reward_from_hits'] for s in epoch_episode_stats])
                r_drops_avg = np.mean([s['reward_from_drops'] for s in epoch_episode_stats])
                r_aim_avg = np.mean([s['reward_from_aim_bonus'] for s in epoch_episode_stats])
                r_penalties_avg = np.mean([s['reward_from_penalties'] for s in epoch_episode_stats])
                r_intent_avg = np.mean([s['reward_from_intent'] for s in epoch_episode_stats])
            else:
                sr = 0.0
                timeout_rate = 0.0
                accuracy = 0.0
                avg_shots = 0.0
                avg_kill_tick = -1
                r_kills_avg = 0.0
                r_hits_avg = 0.0
                r_drops_avg = 0.0
                r_aim_avg = 0.0
                r_penalties_avg = 0.0
                r_intent_avg = 0.0
                kills = 0
                n_episodes = 0
                
            history_sr.append(float(sr))
            history_accuracy.append(float(accuracy))
            
            # Atualização Controlada do Oponente (Self-Play)
            if chance_self_play > 0.0:
                if sr >= 60.0 and (epoch - last_opp_update_epoch) >= 50:
                    print(f"  [Self-Play] Agente dominou o oponente (Win Rate: {sr:.1f}% >= 60%). Atualizando oponente!")
                    opp_params = state.params
                    historical_params_pool.append(state.params)
                    if len(historical_params_pool) > 10:
                        historical_params_pool.pop(0)
                    last_opp_update_epoch = epoch
            else:
                opp_params = state.params
                last_opp_update_epoch = epoch
            
            if epoch % 10 == 0 or epoch == 1:
                cur_alpha = 0.01  # Fixo
                
                # Recompensa Média Suavizada (últimos 100 epochs)
                r_avg100 = np.mean(history_rewards[-100:]) if len(history_rewards) > 0 else epoch_total_reward
                sr_avg100 = np.mean(history_sr[-100:]) if len(history_sr) > 0 else sr
                
                print(f"[Epoch {epoch}/{epochs}] Progress:{curriculum_progress:.3f} | "
                      f"Loss:{loss:.2f} Pi:{pi_loss:.2f} Ent:{ent:.2f} α:{cur_alpha:.3f} | "
                      f"R:{epoch_total_reward:.1f} R100:{r_avg100:.1f} | "
                      f"SR:{sr:.0f}% SR100:{sr_avg100:.0f}% | "
                      f"Acc:{accuracy:.0f}% Shots:{avg_shots:.1f} KillT:{avg_kill_tick:.0f} | "
                      f"V:{v_mean_epoch:.1f} | {fps:.0f} s/s")
                
                # Detalhamento de recompensa (a cada 50 epochs)
                if epoch % 50 == 0:
                    print(f"    R_decomp: kills={r_kills_avg:.1f} hits={r_hits_avg:.1f} "
                          f"drops={r_drops_avg:.1f} aim={r_aim_avg:.2f} intent={r_intent_avg:.2f} penalties={r_penalties_avg:.1f} "
                          f"| Eps:{n_episodes} K:{kills} TO_rate:{timeout_rate:.0f}% Adv:{adv_mean_epoch:.3f}")
                      
                # Log em CSV
                import csv
                csv_file = 'training_metrics.csv'
                file_exists = os.path.exists(csv_file)
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            'Epoch', 'Progress', 'Loss', 'Pi_Loss', 'Entropy', 'Alpha',
                            'R_epoch', 'R_avg100', 'SR', 'SR_avg100',
                            'Accuracy', 'Avg_Shots', 'Avg_Kill_Tick', 'Timeout_Rate',
                            'V_mean', 'Adv_mean',
                            'R_kills', 'R_hits', 'R_drops', 'R_aim', 'R_intent', 'R_penalties',
                            'N_episodes', 'FPS'
                        ])
                    writer.writerow([
                        epoch, f"{curriculum_progress:.4f}",
                        f"{loss:.3f}", f"{pi_loss:.3f}", f"{ent:.3f}", f"{cur_alpha:.4f}",
                        f"{epoch_total_reward:.2f}", f"{r_avg100:.2f}",
                        f"{sr:.2f}", f"{sr_avg100:.2f}",
                        f"{accuracy:.2f}", f"{avg_shots:.1f}", f"{avg_kill_tick:.1f}", f"{timeout_rate:.1f}",
                        f"{v_mean_epoch:.2f}", f"{adv_mean_epoch:.4f}",
                        f"{r_kills_avg:.2f}", f"{r_hits_avg:.2f}", f"{r_drops_avg:.2f}",
                        f"{r_aim_avg:.3f}", f"{r_intent_avg:.2f}", f"{r_penalties_avg:.2f}",
                        n_episodes, f"{fps:.0f}"
                    ])
                
                # --- PROGRESSÃO CONTÍNUA DO CURRÍCULO ---
                # Incrementa curriculum_progress em +0.005 quando recompensa média é positiva
                if epoch_total_reward > 0 and curriculum_progress < 1.0:
                    old_progress = curriculum_progress
                    curriculum_progress = min(1.0, curriculum_progress + 0.005)
                    env.update_curriculum(0.005)
                    
                    # Ancora EWC a cada 0.1 de avanço
                    if int(curriculum_progress * 10) > int(old_progress * 10):
                        ewc_anchor = state.params
                        ewc_lambda = 0.1
                        milestone = curriculum_progress
                        print(f">>> MILESTONE! Progress: {milestone:.3f} | Salvando checkpoint...")
                        with open(f"modelo_progress_{milestone:.2f}.msgpack", "wb") as f:
                            f.write(flax.serialization.to_bytes(state.params))
                        
                # Self-play pool (somente no final do currículo)
                if curriculum_progress >= 0.95 and epoch % 500 == 0:
                    historical_params_pool.append(copy.deepcopy(state.params))
                    if len(historical_params_pool) > 10:
                        historical_params_pool.pop(0)
                        
            if epoch % 10 == 0 or epoch == 1:
                r_avg_plot = [np.mean(history_rewards[max(0, i-100):i+1]) for i in range(len(history_rewards))]
                sr_avg_plot = [np.mean(history_sr[max(0, i-100):i+1]) for i in range(len(history_sr))]
                
                # Plot 1: Recompensas, Sucesso, Acurácia e Progresso
                plt.figure(figsize=(14, 16))
                
                plt.subplot(4, 1, 1)
                plt.plot(history_rewards, label='R_epoch', color='lightgreen', alpha=0.4)
                plt.plot(r_avg_plot, label='R_avg100', color='green', linewidth=2)
                plt.title('Evolução do Treinamento PPO (ResNet + Memory)')
                plt.ylabel('Recompensa')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(4, 1, 2)
                plt.plot(history_sr, label='SR (%)', color='mediumpurple', alpha=0.4)
                plt.plot(sr_avg_plot, label='SR_avg100', color='purple', linewidth=2)
                plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Threshold 70%')
                plt.ylabel('SR (%)')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(4, 1, 3)
                plt.plot(history_accuracy, label='Accuracy (%)', color='orange')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(4, 1, 4)
                plt.plot(history_progress, label='Curriculum Progress', color='magenta')
                plt.ylabel('Progress')
                plt.xlabel('Epochs')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('training_metrics_1.png')
                plt.close()

                # Plot 2: Entropia, Valor, e Loss
                plt.figure(figsize=(14, 12))
                
                plt.subplot(3, 1, 1)
                plt.plot(history_entropy, label='Entropy', color='teal')
                plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='H_TARGET')
                plt.ylabel('Entropy (nats)')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(3, 1, 2)
                plt.plot(history_v_mean, label='V_mean (Critic)', color='coral')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(3, 1, 3)
                plt.plot(history_pi_loss, label='Pi Loss', color='blue', alpha=0.7)
                plt.plot(history_v_loss, label='V Loss', color='red', alpha=0.7)
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('training_metrics_2.png')
                plt.close()
                
    except KeyboardInterrupt:
        print("\n[!] Treinamento interrompido pelo usuário via Ctrl+C.")
    finally:
        print("\nSalvando os pesos (checkpoint) no disco...")
        env.close()
        try:
            # Converte params para CPU (numpy) primeiro para liberar VRAM
            cpu_params = jax.device_get(state.params)
            bytes_output = flax.serialization.to_bytes(cpu_params)
            with open("modelo_treinado.msgpack", "wb") as f:
                f.write(bytes_output)
            print("Salvo com sucesso em: 'modelo_treinado.msgpack'!")
        except Exception as e:
            print(f"[AVISO] Falha ao salvar checkpoint: {e}")
            print("Tentando salvar parâmetro por parâmetro...")
            try:
                cpu_params = jax.tree_util.tree_map(lambda x: np.array(x), state.params)
                bytes_output = flax.serialization.to_bytes(cpu_params)
                with open("modelo_treinado.msgpack", "wb") as f:
                    f.write(bytes_output)
                print("Salvo com sucesso (fallback) em: 'modelo_treinado.msgpack'!")
            except Exception as e2:
                print(f"[ERRO FATAL] Impossível salvar checkpoint: {e2}")

if __name__ == "__main__":
    run_training_loop()
