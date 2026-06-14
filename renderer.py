# pyrefly: ignore [missing-import]
import pygame
import numpy as np
import os
import argparse
from engine import GameEngine

try:
    import jax
    import jax.numpy as jnp
    import flax.serialization
    from train import ActorCritic, NUM_FRAMES, CHANNELS_PER_FRAME
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    NUM_FRAMES = 4
    CHANNELS_PER_FRAME = 5

try:
    from tactical_llm import TacticalLLM
    from audio_provocateur import AudioProvocateur
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Configurações de Tela
CELL_SIZE = 12
MAP_SIZE = 64
WINDOW_SIZE = CELL_SIZE * MAP_SIZE

# Cores
COLOR_BG = (20, 20, 20)
COLOR_WALL = (100, 100, 100)
COLOR_ARENA_BORDER = (60, 180, 60)  # Borda da arena virtual (verde)
COLOR_OUTSIDE_ARENA = (40, 40, 50)  # Área fora da arena
COLOR_PLAYER = (50, 150, 250)
COLOR_ENEMY = (250, 50, 50)
COLOR_PROJ_P = (200, 250, 250)
COLOR_PROJ_FATAL = (255, 0, 0)
COLOR_PROJ_FREEZE = (0, 255, 255)
COLOR_PROJ_POISON = (0, 255, 0)
COLOR_SHIELD = (100, 200, 250)
COLOR_TEXT = (255, 255, 255)

# Cores dos Drops
COLOR_DROP_HP = (255, 100, 100)
COLOR_DROP_FREEZE = (100, 200, 255)
COLOR_DROP_POISON = (100, 255, 100)
COLOR_DROP_SPEED = (255, 255, 100)

class Renderer:
    def __init__(self, model_file="", curriculum_progress=0.0, spectator=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Tensor Mage Arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15)
        self.font_small = pygame.font.SysFont("monospace", 12)
        # Inicializar engine em is_training=True se o Bot for usado para manter simetria perfeita
        self.engine = GameEngine(map_size=MAP_SIZE, num_frames=NUM_FRAMES, is_training=(model_file != ""))
        self.curriculum_progress = curriculum_progress
        self.engine.reset(curriculum_progress=self.curriculum_progress)
        
        # Variáveis do Bot Neural
        self.use_bot = False
        self.spectator = spectator
        self.bot_params = None
        
        if HAS_MODEL and model_file and os.path.exists(model_file):
            print(f"Carregando o Bot Neural treinado: {model_file}")
            model = ActorCritic()
            rng = jax.random.PRNGKey(0)
            dummy_spatial = jnp.zeros((1, MAP_SIZE, MAP_SIZE, NUM_FRAMES * CHANNELS_PER_FRAME))
            dummy_scalar = jnp.zeros((1, NUM_FRAMES, 15))
            dummy_mask = jnp.ones((1, 5), dtype=bool)
            self.bot_params = model.init(rng, dummy_spatial, dummy_scalar, dummy_mask)
            with open(model_file, "rb") as f:
                self.bot_params = flax.serialization.from_bytes(self.bot_params, f.read())
            self.use_bot = True
        
        # --- Sistema LLM Tático + Áudio Provocador ---
        self.tactical_llm = None
        self.audio_worker = None
        self.game_tick = 0
        self.audio_worker = AudioProvocateur(
            event_queue=self.engine.event_queue,
            cooldown_seconds=0
        )
        self.audio_worker.start()
        
    def render(self, state):
        self.screen.fill(COLOR_BG)
        
        # Obter limites da arena diretos da engine
        progress = state.get("curriculum_progress", self.curriculum_progress)
        arena_min, arena_max = self.engine.get_arena_bounds()
        
        # Hitbox visual exata ditada pela engine física
        hitbox_radius = self.engine.lerp(self.engine.hitbox_radius_max, self.engine.hitbox_radius_min, progress)
        
        # 1. Desenhar Paredes, Drops e Arena
        walls = state["walls"]
        drops = state["drops"]
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Área fora da arena (visualizar a jaula)
                if x < int(arena_min) or x >= int(arena_max) or y < int(arena_min) or y >= int(arena_max):
                    if walls[x, y] == 1:
                        pygame.draw.rect(self.screen, COLOR_OUTSIDE_ARENA, rect)
                    else:
                        pygame.draw.rect(self.screen, (30, 30, 35), rect)
                elif walls[x, y] == 1:
                    pygame.draw.rect(self.screen, COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1) # Borda
                elif drops[x, y] > 0:
                    color = COLOR_DROP_HP
                    if drops[x, y] == 2: color = COLOR_DROP_FREEZE
                    elif drops[x, y] == 3: color = COLOR_DROP_POISON
                    elif drops[x, y] == 4: color = COLOR_DROP_SPEED
                    pygame.draw.circle(self.screen, color, (int(x * CELL_SIZE + CELL_SIZE//2), int(y * CELL_SIZE + CELL_SIZE//2)), CELL_SIZE // 3)
        
        # Desenhar borda da arena virtual (linha tracejada verde)
        arena_rect = pygame.Rect(
            int(arena_min * CELL_SIZE), int(arena_min * CELL_SIZE),
            int((arena_max - arena_min) * CELL_SIZE), int((arena_max - arena_min) * CELL_SIZE)
        )
        pygame.draw.rect(self.screen, COLOR_ARENA_BORDER, arena_rect, 2)
                    
        # 2. Desenhar Entidades
        px, py = state["player_pos"]
        ex, ey = state["enemy_pos"]
        
        # Inimigo (hitbox dinâmica)
        pygame.draw.circle(self.screen, COLOR_ENEMY, (int(ex * CELL_SIZE), int(ey * CELL_SIZE)), max(3, int(hitbox_radius * CELL_SIZE * 0.5)))
        # Player
        pygame.draw.circle(self.screen, COLOR_PLAYER, (int(px * CELL_SIZE), int(py * CELL_SIZE)), CELL_SIZE // 2)
        
        # Escudo
        p_stats = state["player_stats"]
        if p_stats[1] > 0:
            pygame.draw.circle(self.screen, COLOR_SHIELD, (int(px * CELL_SIZE), int(py * CELL_SIZE)), CELL_SIZE, 2)
            
        # 3. Desenhar Projéteis
        p_active = state["proj_active"]
        p_pos = state["proj_pos"]
        p_type = state["proj_type"]
        p_owner = state.get("proj_owner", np.zeros(len(p_active))) # Fallback seguro
        
        for i in range(len(p_active)):
            if p_active[i]:
                x, y = p_pos[i]
                if p_owner[i] == 1: # Inimigo
                    color = (255, 150, 50) # Laranja
                else:
                    color = COLOR_PROJ_P
                    if p_type[i] == 1: color = COLOR_PROJ_FATAL
                    elif p_type[i] == 2: color = COLOR_PROJ_FREEZE
                    elif p_type[i] == 3: color = COLOR_PROJ_POISON
                size = 5 if p_type[i] == 1 else 3
                pygame.draw.circle(self.screen, color, (int(x * CELL_SIZE), int(y * CELL_SIZE)), size)
                
        # 4. HUD
        hp_text = self.font.render(f"HP: {p_stats[0]}", True, COLOR_TEXT)
        enemy_hp_text = self.font.render(f"Enemy HP: {state['enemy_stats'][0]:.0f}", True, COLOR_TEXT)
        progress_text = self.font.render(f"Progress: {progress:.3f}", True, (100, 255, 100))
        
        self.screen.blit(hp_text, (10, 10))
        self.screen.blit(enemy_hp_text, (WINDOW_SIZE - 180, 10))
        self.screen.blit(progress_text, (WINDOW_SIZE // 2 - 60, 10))
        
        # Mostra o comando (Intent) do treinamento
        intent_names = [
            "Agressivo: Perseguir", "Flanquear: Esquerda", "Flanquear: Direita",
            "All-In Letal", "Evasivo: Recuar", "Defensivo: Proteger",
            "Movimento Errático", "Sobrevivência Tartaruga", "Controle Territorial",
            "Patrulha Perimetral", "Caçar Recursos"
        ]
        intent_vec = state.get("current_intent")
        if intent_vec is not None and np.any(intent_vec):
            idx = int(np.argmax(intent_vec))
            if 0 <= idx < len(intent_names):
                intent_name = intent_names[idx]
                intent_text = self.font_small.render(f"Tática (Treino): {intent_name}", True, (255, 220, 100))
                self.screen.blit(intent_text, (WINDOW_SIZE // 2 - 80, 28))
        
        # Cooldowns (dividir por 30 para segundos)
        ui_y = 30
        def draw_hud(text, y):
            surface = self.font_small.render(text, True, COLOR_TEXT)
            self.screen.blit(surface, (10, y))
            return y + 15
            
        # Mostrar apenas skills desbloqueadas
        skill_mask = state.get("skill_mask", {})
        
        # Buffs Ativos
        if p_stats[7] > 0: ui_y = draw_hud(f"BUFF: Gelo ({p_stats[7]/30.0:.1f}s)", ui_y)
        if p_stats[8] > 0: ui_y = draw_hud(f"BUFF: Veneno ({p_stats[8]/30.0:.1f}s)", ui_y)
        if p_stats[9] > 0: ui_y = draw_hud(f"BUFF: Vel. ({p_stats[9]/30.0:.1f}s)", ui_y)
        
        # Info da Arena
        arena_info = self.font_small.render(f"Arena: {int(arena_max-arena_min)}x{int(arena_max-arena_min)} | Hitbox: {hitbox_radius:.1f}", True, (150, 150, 150))
        self.screen.blit(arena_info, (10, WINDOW_SIZE - 40))
        
        # Legenda Drops
        leg_y = WINDOW_SIZE - 60
        self.screen.blit(self.font_small.render("Legenda Drops:", True, COLOR_TEXT), (10, leg_y))
        pygame.draw.circle(self.screen, COLOR_DROP_HP, (15, leg_y+20), 4)
        self.screen.blit(self.font_small.render("HP", True, COLOR_TEXT), (25, leg_y+15))
        pygame.draw.circle(self.screen, COLOR_DROP_FREEZE, (60, leg_y+20), 4)
        self.screen.blit(self.font_small.render("Gelo", True, COLOR_TEXT), (70, leg_y+15))
        pygame.draw.circle(self.screen, COLOR_DROP_POISON, (120, leg_y+20), 4)
        self.screen.blit(self.font_small.render("Veneno", True, COLOR_TEXT), (130, leg_y+15))
        pygame.draw.circle(self.screen, COLOR_DROP_SPEED, (190, leg_y+20), 4)
        self.screen.blit(self.font_small.render("Veloc", True, COLOR_TEXT), (200, leg_y+15))
        
        pygame.display.flip()
        
    def get_action_from_input(self):
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        
        dx, dy = 0.0, 0.0
        if keys[pygame.K_w]: dy -= 1.0
        if keys[pygame.K_s]: dy += 1.0
        if keys[pygame.K_a]: dx -= 1.0
        if keys[pygame.K_d]: dx += 1.0
        
        px, py = self.engine.player_pos
        px_screen = px * CELL_SIZE
        py_screen = py * CELL_SIZE
        aim_x = mouse_pos[0] - px_screen
        aim_y = mouse_pos[1] - py_screen
        
        return {
            "move": [dx, dy],
            "aim": [aim_x, aim_y],
            "shoot": mouse_buttons[0] # Botão esquerdo
        }

    def get_intent_from_state(self, state):
        intent = np.zeros(11, dtype=np.float32)
        p_hp = state["player_stats"][0]
        e_hp = state["enemy_stats"][0]
        dist = np.linalg.norm(state["player_pos"] - state["enemy_pos"])
        
        if e_hp <= 10: intent[3] = 1.0
        elif p_hp < 30: intent[4] = 1.0
        elif e_hp < 50 and dist > 15: intent[0] = 1.0
        elif dist < 8: intent[6] = 1.0
        elif dist > 30: intent[0] = 1.0
        else: intent[8] = 1.0
            
        if p_hp > 50:
            drop_positions = np.argwhere(state.get("drops", np.zeros((64, 64))) == 1) if "drops" in state else []
            if len(drop_positions) > 0:
                dists_to_drops = np.linalg.norm(drop_positions - state["player_pos"], axis=1)
                if np.min(dists_to_drops) < 8.0:
                    intent = np.zeros(11, dtype=np.float32)
                    intent[10] = 1.0
        return intent

    def run(self):
        state = self.engine.get_state()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            action = self.get_action_from_input()
            
            # Recalcula a intenção localmente para exibir na tela
            self.engine.current_intent = self.get_intent_from_state(state)
            self.game_tick += 1
            
            # ---------------------------------------------
            # Lógica de Inferência do Bot
            # ---------------------------------------------
            enemy_action = None
            if self.use_bot and self.bot_params is not None:
                if self.spectator:
                    # Bot controla o Player (Azul)
                    obs = state
                    spatial = obs["spatial_obs"][np.newaxis, ...]
                    scalars = obs["scalar_obs"][np.newaxis, ...]
                    mask = obs["action_mask"][np.newaxis, ...]
                    
                    model = ActorCritic()
                    logits, _ = model.apply(self.bot_params, jnp.array(spatial), jnp.array(scalars), jnp.array(mask))
                    
                    move_idx = int(np.argmax(np.array(logits["move"][0])))
                    aim = np.array(logits["aim"][0])
                    shoot = (np.argmax(np.array(logits["shoot"][0])) == 1)
                    
                    action["move_idx"] = move_idx
                    action["aim"] = aim
                    action["shoot"] = shoot
                else:
                    # Bot controla o Inimigo (Vermelho) - Frame Inversion
                    spatial = state["spatial_obs"].copy()
                    # Inverter os canais de Player(2) e Enemy(3) nos frames empilhados
                    for frame in range(NUM_FRAMES):
                        base_idx = frame * CHANNELS_PER_FRAME
                        temp = spatial[:, :, base_idx + 2].copy()
                        spatial[:, :, base_idx + 2] = spatial[:, :, base_idx + 3]
                        spatial[:, :, base_idx + 3] = temp
                    
                    spatial = np.expand_dims(spatial, axis=0)
                        
                    scalars = np.zeros((1, NUM_FRAMES, 15), dtype=np.float32)
                    pad_enemy = np.zeros(10, dtype=np.float32)
                    pad_enemy[:3] = state["enemy_stats"][:3]
                    
                    to_player = state["player_pos"] - state["enemy_pos"]
                    norm_e = np.linalg.norm(to_player)
                    enemy_aim = to_player / norm_e if norm_e > 0 else np.array([0.0, 1.0], dtype=np.float32)
                    
                    e_scalar = np.concatenate([pad_enemy, state["player_stats"][:3], enemy_aim]).astype(np.float32)
                    scalars[0, :, :] = e_scalar
                    
                    mask = np.ones((1, 5), dtype=bool)
                    ex, ey = state["enemy_pos"]
                    dirs = [[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]]
                    for i in range(1, 5):
                        nx, ny = ex + dirs[i][0], ey + dirs[i][1]
                        if nx < 1 or nx >= MAP_SIZE-1 or ny < 1 or ny >= MAP_SIZE-1:
                            mask[0, i] = False
                        elif state["walls"][int(nx), int(ny)]:
                            mask[0, i] = False
                    
                    model = ActorCritic()
                    logits, _ = model.apply(self.bot_params, jnp.array(spatial), jnp.array(scalars), jnp.array(mask))
                    
                    move_idx = int(np.argmax(np.array(logits["move"][0])))
                    aim = np.array(logits["aim"][0])
                    shoot = (np.argmax(np.array(logits["shoot"][0])) == 1)
                    
                    enemy_action = {"move_idx": move_idx, "aim": aim, "shoot": shoot}

            state, reward, done, stats = self.engine.step(action, enemy_action=enemy_action, curriculum_progress=self.curriculum_progress)
            
            if done:
                print(f"Game Over: {stats.get('episode_result', '???')}")
                self.engine.reset(curriculum_progress=self.curriculum_progress)
                state = self.engine.get_state()
            self.render(state)
                
            self.clock.tick(30) # 30 FPS
            
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renderer for Tensor Mage Arena")
    parser.add_argument("--model", type=str, default="", help="Caminho para o modelo (ex: modelo_progress_0.50.msgpack)")
    parser.add_argument("--progress", type=float, default=0.0, help="Progresso do currículo (0.0 a 1.0)")
    parser.add_argument("--play", action="store_true", help="Se informado com --model, você joga (Azul) contra o bot (Vermelho)")
    args = parser.parse_args()
    
    spectator_mode = True if not args.play else False
    Renderer(model_file=args.model, curriculum_progress=args.progress, spectator=spectator_mode).run()
