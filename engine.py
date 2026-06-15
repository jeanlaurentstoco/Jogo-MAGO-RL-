import numpy as np
import heapq
import queue

class GameEngine:
    def __init__(self, map_size=64, max_projectiles=128, headless=True, num_frames=16, is_training=False):
        self.map_size = map_size
        self.max_projectiles = max_projectiles
        self.headless = headless
        self.num_frames = num_frames
        self.is_training = is_training
        self.channels_per_frame = 5
        
        # Action Map: 0: Stay, 1: North (0, -1), 2: South (0, 1), 3: East (1, 0), 4: West (-1, 0)
        self.move_dirs = np.array([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]], dtype=np.float32)
        
        # =============================================
        # CURRÍCULO CONTÍNUO (Continuous Curriculum)
        # =============================================
        self.curriculum_progress = 0.0  # Float 0.0 → 1.0
        
        # --- LIMITES FÍSICOS (LERP) ---
        # Hitbox: começa colossal (fácil acertar) e encolhe até realista
        self.player_speed_base = 0.5    # Velocidade base do player (referência)
        
        if self.is_training:
            self.enemy_hp_min = 300.0
            self.enemy_hp_max = 300.0
            self.player_hp_base = 300.0
        else:
            self.enemy_hp_min = 300.0       # Em progress=0.0
            self.enemy_hp_max = 200.0       # Em progress=1.0
            self.player_hp_base = 200.0
            
        # Velocidade do Inimigo: parado → rápido (Igual ao player em p=1.0)
        self.enemy_speed_min = 0.15
        self.enemy_speed_max = 0.6
        
        # Velocidade dos Projéteis do Inimigo (Igual ao player em p=1.0)
        self.enemy_proj_speed_min = 0.8
        self.enemy_proj_speed_max = 1.5
        
        # Intervalo de Tiro do Inimigo: lento → rápido (Igual ao player em p=1.0)
        self.enemy_shoot_interval_min = 10
        self.enemy_shoot_interval_max = 3
        
        # Hitbox do Inimigo: colossal → realista (Igual ao player em p=1.0)
        self.hitbox_radius_max = 6
        self.hitbox_radius_min = 1.0
        
        # --- ARENA DINÂMICA (Jaula de Vidro) ---
        # A observação NUNCA muda de shape. A arena jogável cresce.
        self.arena_size_min = 60        # Arena expandida no início para dar espaço de desvio
        self.arena_size_max = self.map_size - 2  # Arena total (62x62 dentro das bordas)
        
        # --- LIMIARES DE DESBLOQUEIO (Threshold Masking) ---
        self.threshold_movement = 0.00   # Movimento liberado
        self.threshold_bounce = 0.40     # Ricochete de projéteis ativado
        self.threshold_maze_walls = 0.20  # Paredes internas aparecem
        
        # --- LIMIARES DE IA INIMIGA ---
        self.threshold_enemy_ai = 0.00    # Inimigo começa a se mover
        self.threshold_enemy_shoot = 0.00 # Inimigo começa a atirar
        self.threshold_enemy_astar = 0.65  # Inimigo usa A* pathfinding
        
        # --- SISTEMA DE INTENÇÕES TÁTICAS (FSM de 6 estados) ---
        # 0: ATACAR, 1: EXECUTAR, 2: DEFENDER, 3: FUGIR, 4: ENCURRALAR, 5: CONTROLAR_MAPA
        self.num_intents = 6
        self.current_intent = np.zeros(self.num_intents, dtype=np.float32)
        self.intent_scale = 5.0  # Multiplicador global inflacionado para sobrepujar a recompensa de combate bruto
        self.event_queue = queue.Queue()  # Fila thread-safe para eventos de gameplay
        
        self.reset()
        
    # =============================================
    # INTERPOLAÇÃO LINEAR (LERP)
    # =============================================
    def lerp(self, min_val, max_val, t=None):
        """Interpolação linear: min + (max - min) * t"""
        if t is None:
            t = self.curriculum_progress
        return min_val + (max_val - min_val) * t
    
    # =============================================
    # ATUALIZAÇÃO DO PROGRESSO
    # =============================================
    def update_curriculum(self, increment: float):
        """Atualiza o progresso do currículo. Chamado pelo loop de treino.
        
        Args:
            increment: Valor microscópico (ex: +0.001) a ser somado ao progresso.
        """
        self.curriculum_progress = float(np.clip(
            self.curriculum_progress + increment, 0.0, 1.0
        ))
        
    def reset(self, curriculum_progress=None):
        if curriculum_progress is not None:
            self.curriculum_progress = float(np.clip(curriculum_progress, 0.0, 1.0))
        
        p = self.curriculum_progress
        
        self.visited_coords = np.zeros((self.map_size, self.map_size), dtype=bool)
        self.map_drops = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        
        # O HP base depende se estamos no treino ou jogo normal
        self.player_stats = np.array([self.player_hp_base, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # HP do Inimigo via LERP
        enemy_hp = self.lerp(self.enemy_hp_min, self.enemy_hp_max, p)
        self.enemy_stats = np.array([enemy_hp, 0, 0], dtype=np.float32)
        
        self.proj_active = np.zeros(self.max_projectiles, dtype=bool)
        self.proj_pos = np.zeros((self.max_projectiles, 2), dtype=np.float32)
        self.proj_vel = np.zeros((self.max_projectiles, 2), dtype=np.float32)
        self.proj_bounces = np.zeros(self.max_projectiles, dtype=np.int32)
        self.proj_type = np.zeros(self.max_projectiles, dtype=np.int32)
        self.proj_owner = np.zeros(self.max_projectiles, dtype=np.int32) # 0 = Player, 1 = Enemy
        self.proj_cooldown = 0
        self.enemy_proj_cooldown = 0
        
        self.frame_buffer = np.zeros((self.map_size, self.map_size, self.num_frames * self.channels_per_frame), dtype=np.float32)
        self.scalar_buffer = np.zeros((self.num_frames, 15), dtype=np.float32)
        
        self.pos_history = np.zeros((20, 2), dtype=np.float32)
        self.tick = 0
        self.last_power_tick = 0
        self.shots_fired = 0
        self.last_aim_vec = None
        self.has_rotated_aim = False
        self.consecutive_hits = 0
        self.fatal_bounce = False
        self.shot_this_tick = False
        
        # --- Contadores de Diagnóstico (Instrumentação) ---
        self.shots_hit = 0
        self.kill_tick = -1
        self.episode_result = 'ongoing'  # 'kill', 'timeout', 'death'
        self.reward_from_kills = 0.0
        self.reward_from_hits = 0.0
        self.reward_from_drops = 0.0
        self.reward_from_aim_bonus = 0.0
        self.reward_from_penalties = 0.0
        self.reward_from_intent = 0.0
        
        # --- Contadores de Anomalia (Event Queue) ---
        self.enemy_miss_streak = 0
        self.player_hit_streak = 0
        self.prev_dist_to_enemy = None
        self.prev_enemy_wall_dist = None
        self.prev_move_dir = None
        
        self.player_idle_ticks = 0
        self.player_flee_ticks = 0
        self.player_cornered_ticks = 0
        self.consecutive_hits = 0
        self.last_hit_tick = 0
        self.death_reason = "em combate"
        
        # Esvazia a fila antiga (se existir) para evitar eco de vidas passadas
        if hasattr(self, 'event_queue'):
            while not self.event_queue.empty():
                try: self.event_queue.get_nowait()
                except queue.Empty: break
        
        self._spawn_entities()
            
        self.generate_walls()
        
        # Drop inicial próximo para progress baixo
        if p < 0.15:
            self._spawn_drop_at_distance(3.0)
            
        for _ in range(4):
            self.push_frame()
            
        return self.get_state()

    # =============================================
    # ARENA DINÂMICA (Jaula de Vidro)
    # =============================================
    def get_arena_bounds(self):
        """Retorna (arena_min, arena_max) — os limites jogáveis baseados no progress.
        
        A arena é centrada no grid e cresce com o curriculum_progress.
        """
        current_arena = self.lerp(self.arena_size_min, self.arena_size_max)
        center = self.map_size / 2.0
        half = current_arena / 2.0
        arena_min = max(1.0, center - half)
        arena_max = min(float(self.map_size - 1), center + half)
        return arena_min, arena_max
    
    def clamp_to_arena(self, pos):
        """Confina uma posição dentro da arena jogável."""
        arena_min, arena_max = self.get_arena_bounds()
        pos[0] = np.clip(pos[0], arena_min, arena_max - 1)
        pos[1] = np.clip(pos[1], arena_min, arena_max - 1)
        return pos
    
    def is_inside_arena(self, x, y):
        """Verifica se uma posição está dentro da arena jogável."""
        arena_min, arena_max = self.get_arena_bounds()
        return (arena_min <= x < arena_max) and (arena_min <= y < arena_max)

    def _spawn_drop_at_distance(self, target_dist):
        self.map_drops.fill(0)
        arena_min, arena_max = self.get_arena_bounds()
        amin, amax = int(arena_min), int(arena_max)
        
        candidates = []
        for x in range(amin, amax):
            for y in range(amin, amax):
                if not self.map_walls[x, y]:
                    dist = np.linalg.norm(np.array([x, y]) - self.player_pos)
                    if 2.0 <= dist <= 4.0:
                        candidates.append((x, y))
        if candidates:
            cx, cy = candidates[np.random.randint(len(candidates))]
            self.map_drops[cx, cy] = 1
        else:
            rx = np.random.randint(amin, max(amin + 1, amax))
            ry = np.random.randint(amin, max(amin + 1, amax))
            if not self.map_walls[rx, ry]:
                self.map_drops[rx, ry] = 1

    def _spawn_entities(self):
        """Spawn de player e inimigo confinados dentro da arena jogável."""
        arena_min, arena_max = self.get_arena_bounds()
        amin, amax = int(arena_min), int(arena_max)
        
        # Garantir que a arena tenha espaço mínimo
        if amax - amin < 4:
            amax = amin + 4
            
        # Spawn totalmente aleatório, mas garantindo que nasçam longe um do outro
        # Exigência: devem nascer a pelo menos 70% da extensão da arena de distância.
        min_dist = (amax - amin) * 0.7
        
        attempts = 0
        while True:
            px = np.random.randint(amin, amax)
            py = np.random.randint(amin, amax)
            ex = np.random.randint(amin, amax)
            ey = np.random.randint(amin, amax)
            attempts += 1
            
            # Se tentou muitas vezes e não achou, reduz a exigência de distância para não travar
            if attempts > 100:
                min_dist *= 0.9
                attempts = 0
                
            if np.linalg.norm(np.array([px, py]) - np.array([ex, ey])) >= min_dist:
                self.player_pos = np.array([px, py], dtype=np.float32)
                self.enemy_pos = np.array([ex, ey], dtype=np.float32)
                break

    def _bsp_split(self, x, y, w, h, depth, max_depth, rooms):
        """Divide recursivamente o espaço em sub-retângulos via BSP.
        
        Args:
            x, y: Canto superior esquerdo da partição
            w, h: Largura e altura da partição
            depth: Profundidade atual da recursão
            max_depth: Profundidade máxima (controla número de cômodos)
            rooms: Lista para acumular os cômodos gerados
        """
        MIN_ROOM_SIZE = 8  # Tamanho mínimo de um cômodo
        
        if depth >= max_depth or w < MIN_ROOM_SIZE * 2 or h < MIN_ROOM_SIZE * 2:
            # Folha: cria um cômodo com margem interna
            margin = 1
            room_x = x + margin
            room_y = y + margin
            room_w = max(MIN_ROOM_SIZE - 2, w - margin * 2)
            room_h = max(MIN_ROOM_SIZE - 2, h - margin * 2)
            rooms.append((room_x, room_y, room_w, room_h))
            return
        
        # Decide se divide horizontal ou vertical
        if w > h:
            split_vertical = True
        elif h > w:
            split_vertical = False
        else:
            split_vertical = np.random.rand() > 0.5
        
        if split_vertical:
            # Divide verticalmente (ao longo de X)
            split = np.random.randint(MIN_ROOM_SIZE, max(MIN_ROOM_SIZE + 1, w - MIN_ROOM_SIZE))
            self._bsp_split(x, y, split, h, depth + 1, max_depth, rooms)
            self._bsp_split(x + split, y, w - split, h, depth + 1, max_depth, rooms)
        else:
            # Divide horizontalmente (ao longo de Y)
            split = np.random.randint(MIN_ROOM_SIZE, max(MIN_ROOM_SIZE + 1, h - MIN_ROOM_SIZE))
            self._bsp_split(x, y, w, split, depth + 1, max_depth, rooms)
            self._bsp_split(x, y + split, w, h - split, depth + 1, max_depth, rooms)

    def generate_walls(self):
        """Gera paredes usando BSP (Binary Space Partitioning).
        
        Cria cômodos reais conectados por corredores em vez de blocos aleatórios.
        A complexidade escala com curriculum_progress:
        - progress < 0.20: Arena aberta sem paredes internas
        - progress 0.20-1.0: Cômodos BSP aparecem gradualmente (2-3 níveis de profundidade)
        """
        self.map_walls = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        
        # Bordas externas do grid (sempre presentes)
        self.map_walls[0, :] = 1
        self.map_walls[-1, :] = 1
        self.map_walls[:, 0] = 1
        self.map_walls[:, -1] = 1
        
        # Paredes da Arena Virtual (Jaula de Vidro)
        arena_min, arena_max = self.get_arena_bounds()
        amin, amax = int(np.floor(arena_min)), int(np.ceil(arena_max))
        
        # Preencher tudo fora da arena como parede
        for x in range(self.map_size):
            for y in range(self.map_size):
                if x < amin or x >= amax or y < amin or y >= amax:
                    self.map_walls[x, y] = 1
        
        # Cômodos BSP — aparecem gradualmente após threshold
        p = self.curriculum_progress
        if p > self.threshold_maze_walls:
            maze_t = (p - self.threshold_maze_walls) / (1.0 - self.threshold_maze_walls)
            
            # Profundidade BSP: 1 nível (2 salas) a 3 níveis (4-6 salas)
            bsp_depth = max(1, int(maze_t * 3))
            
            # Área jogável
            arena_w = amax - amin
            arena_h = amax - amin
            
            # Gera cômodos via BSP
            rooms = []
            self._bsp_split(amin, amin, arena_w, arena_h, 0, bsp_depth, rooms)
            
            if len(rooms) >= 2:
                # Passo 1: Preenche a arena inteira com parede
                self.map_walls[amin:amax, amin:amax] = 1
                
                # Passo 2: Escava cada cômodo (interior aberto)
                for (rx, ry, rw, rh) in rooms:
                    x1 = max(amin, rx)
                    y1 = max(amin, ry)
                    x2 = min(amax, rx + rw)
                    y2 = min(amax, ry + rh)
                    if x2 > x1 and y2 > y1:
                        self.map_walls[x1:x2, y1:y2] = 0
                
                # Passo 3: Conecta cômodos adjacentes com corredores de 3 blocos
                for i in range(len(rooms) - 1):
                    r1 = rooms[i]
                    r2 = rooms[i + 1]
                    # Centro de cada cômodo
                    cx1 = r1[0] + r1[2] // 2
                    cy1 = r1[1] + r1[3] // 2
                    cx2 = r2[0] + r2[2] // 2
                    cy2 = r2[1] + r2[3] // 2
                    
                    # Corredor em L: horizontal depois vertical
                    corridor_width = 3
                    half_w = corridor_width // 2
                    
                    # Segmento horizontal
                    x_start = min(cx1, cx2)
                    x_end = max(cx1, cx2) + 1
                    for dy in range(-half_w, half_w + 1):
                        y_cor = np.clip(cy1 + dy, amin, amax - 1)
                        self.map_walls[x_start:x_end, y_cor] = 0
                    
                    # Segmento vertical
                    y_start = min(cy1, cy2)
                    y_end = max(cy1, cy2) + 1
                    for dx in range(-half_w, half_w + 1):
                        x_cor = np.clip(cx2 + dx, amin, amax - 1)
                        self.map_walls[x_cor, y_start:y_end] = 0
                
                # Também conectar o último ao primeiro para garantir circularidade
                if len(rooms) > 2:
                    r1 = rooms[-1]
                    r2 = rooms[0]
                    cx1 = r1[0] + r1[2] // 2
                    cy1 = r1[1] + r1[3] // 2
                    cx2 = r2[0] + r2[2] // 2
                    cy2 = r2[1] + r2[3] // 2
                    corridor_width = 3
                    half_w = corridor_width // 2
                    x_start = min(cx1, cx2)
                    x_end = max(cx1, cx2) + 1
                    for dy in range(-half_w, half_w + 1):
                        y_cor = np.clip(cy1 + dy, amin, amax - 1)
                        self.map_walls[x_start:x_end, y_cor] = 0
                    y_start = min(cy1, cy2)
                    y_end = max(cy1, cy2) + 1
                    for dx in range(-half_w, half_w + 1):
                        x_cor = np.clip(cx2 + dx, amin, amax - 1)
                        self.map_walls[x_cor, y_start:y_end] = 0
            
            # Limpar área ao redor do spawn (raio 5) para garantir jogabilidade
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            ex, ey = int(self.enemy_pos[0]), int(self.enemy_pos[1])
            spawn_radius = 5
            self.map_walls[max(amin, px-spawn_radius):min(amax, px+spawn_radius+1), 
                          max(amin, py-spawn_radius):min(amax, py+spawn_radius+1)] = 0
            self.map_walls[max(amin, ex-spawn_radius):min(amax, ex+spawn_radius+1), 
                          max(amin, ey-spawn_radius):min(amax, ey+spawn_radius+1)] = 0
            
            # Restaurar bordas externas (podem ter sido limpas)
            self.map_walls[0, :] = 1
            self.map_walls[-1, :] = 1
            self.map_walls[:, 0] = 1
            self.map_walls[:, -1] = 1
            
            # Restaurar paredes fora da arena
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if x < amin or x >= amax or y < amin or y >= amax:
                        self.map_walls[x, y] = 1

    def get_spatial_obs(self):
        spatial = np.zeros((self.map_size, self.map_size, 5), dtype=np.float32)
        spatial[:, :, 0] = self.map_walls
        spatial[:, :, 1] = self.map_drops
        
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        if 0 <= px < self.map_size and 0 <= py < self.map_size:
            spatial[px, py, 2] = 1.0
            
        ex, ey = int(self.enemy_pos[0]), int(self.enemy_pos[1])
        if 0 <= ex < self.map_size and 0 <= ey < self.map_size:
            spatial[ex, ey, 3] = 1.0
            
        active = self.proj_active
        pos = self.proj_pos[active].astype(np.int32)
        for i in range(len(pos)):
            pr_x, pr_y = pos[i]
            if 0 <= pr_x < self.map_size and 0 <= pr_y < self.map_size:
                spatial[pr_x, pr_y, 4] += 1.0
                
        return spatial

    def push_frame(self):
        new_frame = self.get_spatial_obs()
        cpf = self.channels_per_frame
        self.frame_buffer[:, :, :-cpf] = self.frame_buffer[:, :, cpf:]
        self.frame_buffer[:, :, -cpf:] = new_frame
        
        to_enemy = self.enemy_pos - self.player_pos
        norm = np.linalg.norm(to_enemy)
        if norm > 0:
            ideal_aim = to_enemy / norm
        else:
            ideal_aim = np.array([0.0, 1.0], dtype=np.float32)
            
        new_scalar = np.concatenate([self.player_stats, self.enemy_stats, ideal_aim]).astype(np.float32)
        self.scalar_buffer[:-1, :] = self.scalar_buffer[1:, :]
        self.scalar_buffer[-1, :] = new_scalar

    # =============================================
    # THRESHOLD MASKING (Desbloqueio por Limiar)
    # =============================================
    def get_action_mask(self):
        """Máscara de movimento (5 direções). Bloqueia movimentação 
        até progress > threshold_movement, além de colisões com paredes/arena."""
        mask = np.ones(5, dtype=bool)
        p = self.curriculum_progress
        
        # Movimento bloqueado (fase torreta) se progress muito baixo
        if p < self.threshold_movement:
            mask[1:] = False
            return mask
        
        noclip = self.player_stats[1] > 0
        px, py = self.player_pos
        arena_min, arena_max = self.get_arena_bounds()
        
        for i in range(1, 5):
            dx, dy = self.move_dirs[i]
            nx, ny = px + dx, py + dy
            # Colisão com bordas da arena (Jaula de Vidro)
            if nx < arena_min or nx >= arena_max or ny < arena_min or ny >= arena_max:
                mask[i] = False
            elif not noclip and self.map_walls[int(nx), int(ny)]:
                mask[i] = False
        return mask
    
    def get_skill_mask(self):
        """Máscara de habilidades discretas baseada no curriculum_progress.
        
        Retorna um dict com booleans indicando se cada skill está desbloqueada.
        Conceito: quando a máscara libera uma skill, a ameaça física (LERP)
        já está num nível onde a skill começa a ser matematicamente vantajosa.
        """
        p = self.curriculum_progress
        return {
            "shoot": True,  # Sempre disponível
        }

    def check_los(self):
        x0, y0 = int(self.player_pos[0]), int(self.player_pos[1])
        x1, y1 = int(self.enemy_pos[0]), int(self.enemy_pos[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.map_walls[x, y]: return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.map_walls[x, y]: return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return True

    def compute_pathfinding(self, start, end):
        """A* Algorithm modificado utilizando Cost Map para bot heurístico."""
        cost_map = np.ones((self.map_size, self.map_size), dtype=np.float32)
        
        # Penaliza paredes e aplica inflação em blocos adjacentes (Custo 2.0)
        for x in range(1, self.map_size-1):
            for y in range(1, self.map_size-1):
                if self.map_walls[x, y]:
                    cost_map[x, y] = np.inf
                elif (self.map_walls[x-1, y] or self.map_walls[x+1, y] or 
                      self.map_walls[x, y-1] or self.map_walls[x, y+1]):
                    cost_map[x, y] = 2.0 # Evita que o bot raspe nas quinas e trave
                    
        # Injeta custo extremo (100.0) na trajetória ativa de projéteis
        active = self.proj_active
        for i in np.where(active)[0]:
            px, py = self.proj_pos[i]
            vx, vy = self.proj_vel[i]
            
            # Projeta nos próximos 3 blocos (incluindo o bloco atual)
            for step in range(4):
                tx, ty = int(px + vx * step), int(py + vy * step)
                if 0 <= tx < self.map_size and 0 <= ty < self.map_size:
                    if cost_map[tx, ty] != np.inf:
                        cost_map[tx, ty] += 100.0

        start_int = (int(start[0]), int(start[1]))
        end_int = (int(end[0]), int(end[1]))
        
        if cost_map[end_int[0], end_int[1]] == np.inf:
            return start # Destino inatingível (preso na parede)
            
        # Algoritmo A-Star com HeapQ
        open_set = []
        heapq.heappush(open_set, (0, start_int))
        came_from = {}
        g_score = {start_int: 0}
        
        # Direções de movimento (8 eixos)
        dirs = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end_int: 
                break
                
            cx, cy = current
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if cost_map[nx, ny] == np.inf: 
                        continue
                        
                    move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    tentative_g = g_score[current] + cost_map[nx, ny] * move_cost
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        came_from[(nx, ny)] = current
                        
                        f_score = tentative_g + np.linalg.norm(np.array([nx, ny]) - np.array(end_int))
                        heapq.heappush(open_set, (f_score, (nx, ny)))
                        
        # Backtracking
        curr = end_int
        path = []
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
            
        if len(path) > 0:
            next_step = path[-1]
            return np.array([next_step[0], next_step[1]], dtype=np.float32)
            
        return start

    def get_state(self):
        return {
            "spatial_obs": self.frame_buffer.copy(),
            "scalar_obs": self.scalar_buffer.copy(),
            "action_mask": self.get_action_mask(),
            "skill_mask": self.get_skill_mask(),
            
            "walls": self.map_walls.copy(),
            "drops": self.map_drops.copy(),
            "player_pos": self.player_pos.copy(),
            "enemy_pos": self.enemy_pos.copy(),
            "player_stats": self.player_stats.copy(),
            "enemy_stats": self.enemy_stats.copy(),
            "proj_active": self.proj_active.copy(),
            "proj_pos": self.proj_pos.copy(),
            "proj_vel": self.proj_vel.copy(),
            "proj_type": self.proj_type.copy(),
            "proj_owner": self.proj_owner.copy(),
            "curriculum_progress": self.curriculum_progress,
            "current_intent": self.current_intent.copy() if hasattr(self, 'current_intent') else np.zeros(6, dtype=np.float32)
        }
    
    def get_episode_stats(self):
        """Retorna estatísticas do episódio atual para instrumentação."""
        accuracy = (self.shots_hit / max(1, self.shots_fired)) * 100.0
        return {
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "accuracy": accuracy,
            "kill_tick": self.kill_tick,
            "episode_result": self.episode_result,
            "ticks_survived": self.tick,
            "reward_from_kills": self.reward_from_kills,
            "reward_from_hits": self.reward_from_hits,
            "reward_from_drops": self.reward_from_drops,
            "reward_from_aim_bonus": self.reward_from_aim_bonus,
            "reward_from_penalties": self.reward_from_penalties,
            "reward_from_intent": self.reward_from_intent,
            "curriculum_progress": self.curriculum_progress,
        }
        
    def update_rewards(self):
        """Calcula e retorna as recompensas/penalidades do tick atual (Reward Shaping).
        
        Adapta penalidades ao curriculum_progress ao invés de curriculum_level.
        """
        step_reward = 0.0
        p = self.curriculum_progress
        
        px_int, py_int = int(self.player_pos[0]), int(self.player_pos[1])
        
        # 1. Penalidade de Tempo (força pressa) — escala com progress
        time_penalty = self.lerp(0.005, 0.01, p)
        step_reward -= time_penalty
        self.reward_from_penalties -= time_penalty
        
        # Atualiza o buffer de histórico (janela de 20 ticks)
        self.pos_history = np.roll(self.pos_history, shift=-1, axis=0)
        self.pos_history[-1] = self.player_pos.copy()
        
        # 2. Penalidade de Campismo (Inércia e Mode Collapse) — ativa gradualmente
        if self.tick >= 20:
            pos_historica = self.pos_history[0]
            # Mede distância Euclidiana para a posição de 20 ticks atrás
            if np.linalg.norm(self.player_pos - pos_historica) < 2.0:
                camp_penalty = self.lerp(0.01, 0.05, p)
                step_reward -= camp_penalty
                self.reward_from_penalties -= camp_penalty
                
        self.tick += 1
        
        
        # 3. Recompensa de Coleta (Sinal SECUNDÁRIO)
        if self.map_drops[px_int, py_int] == 1:
            self.map_drops[px_int, py_int] = 0 # Consome o item
            step_reward += 0.3
            self.reward_from_drops += 0.3
            
            self.player_stats[0] = min(self.player_hp_base, self.player_stats[0] + 20)
            
            if p < 0.15:
                self._spawn_drop_at_distance(3.0)
            
        return step_reward
    
    def compute_intent_reward(self):
        """Calcula recompensa tática via FSM de 6 estados.
        
        Estados:
            0: ATACAR      — Avançar, manter distância de tiro (8-15), atirar agressivamente
            1: EXECUTAR    — All-in: perseguir até melee, não parar de atirar
            2: DEFENDER    — Manter distância > 15, buscar cobertura, atirar quando seguro
            3: FUGIR       — Maximizar distância, ignorar tiro, buscar HP drops
            4: ENCURRALAR  — Posicionar para empurrar inimigo contra parede/canto
            5: CONTROLAR_MAPA — Posição central, patrulhar, coletar drops
        """
        if not np.any(self.current_intent > 0):
            return 0.0
            
        if self.curriculum_progress < self.threshold_movement:
            return 0.0
        
        rewards = np.zeros(self.num_intents, dtype=np.float32)
        
        # --- MÉTRICAS COMPUTADAS UMA VEZ ---
        dist_to_enemy = np.linalg.norm(self.player_pos - self.enemy_pos)
        arena_min, arena_max = self.get_arena_bounds()
        arena_center = (arena_min + arena_max) / 2.0
        dist_to_center = np.linalg.norm(self.player_pos - np.array([arena_center, arena_center]))
        
        px, py = self.player_pos
        ex, ey = self.enemy_pos
        player_wall_dist = min(px - arena_min, arena_max - px, py - arena_min, arena_max - py)
        enemy_wall_dist = min(ex - arena_min, arena_max - ex, ey - arena_min, arena_max - ey)
        
        p_hp = self.player_stats[0]
        e_hp = self.enemy_stats[0]
        p_hp_pct = p_hp / max(1.0, self.player_hp_base)
        
        has_los = self.check_los()
        
        # Deltas (requerem tick anterior)
        dist_delta = 0.0
        if self.prev_dist_to_enemy is not None:
            dist_delta = dist_to_enemy - self.prev_dist_to_enemy
        
        wall_delta = 0.0
        if self.prev_enemy_wall_dist is not None:
            wall_delta = enemy_wall_dist - self.prev_enemy_wall_dist
        
        # --- EVENTOS DE COMPORTAMENTO ---
        def _dispatch_event(ev_type, **kwargs):
            ev = {
                "type": ev_type,
                "tick": self.tick,
                "player_hp": p_hp,
                "enemy_hp": e_hp,
                "distance": dist_to_enemy
            }
            ev.update(kwargs)
            try: self.event_queue.put_nowait(ev)
            except queue.Full: pass

        # Jogador Parado
        if self.prev_move_dir is not None and np.linalg.norm(self.prev_move_dir) < 0.1:
            self.player_idle_ticks += 1
            if self.player_idle_ticks == 60:
                _dispatch_event("player_idle")
        else:
            self.player_idle_ticks = 0
            
        # Jogador Fugindo
        fleeing = False
        if self.prev_move_dir is not None and np.linalg.norm(self.prev_move_dir) > 0.1:
            to_enemy_dir = self.enemy_pos - self.player_pos
            norm_e = np.linalg.norm(to_enemy_dir)
            if norm_e > 0:
                dot = np.dot(self.prev_move_dir, to_enemy_dir / norm_e)
                if dot < -0.5:
                    fleeing = True

        if fleeing:
            self.player_flee_ticks += 1
            if self.player_flee_ticks == 45:  # 1.5 segundo fugindo
                self.death_reason = "tentando fugir das minhas chamas"
                _dispatch_event("player_fleeing")
        else:
            self.player_flee_ticks = 0
            
        # Jogador Encurralado
        if player_wall_dist < 2.0 and dist_to_enemy < 8.0:
            self.player_cornered_ticks += 1
            if self.player_cornered_ticks == 30:  # 1 segundo encurralado
                self.death_reason = "encurralado contra a parede implorando por piedade"
                _dispatch_event("player_cornered")
        else:
            self.player_cornered_ticks = 0
            
        if self.player_flee_ticks == 0 and self.player_cornered_ticks == 0:
            self.death_reason = "em combate franco"
        
        # =============================================
        # FSM DE 6 ESTADOS — RECOMPENSAS ESPECÍFICAS
        # =============================================
        
        # 0: ATACAR — Avançar + manter distância ideal (8-15) + atirar
        # Recompensa: reduzir distância até range ideal, atirar com LOS
        if 8.0 <= dist_to_enemy <= 15.0:
            rewards[0] = 0.15  # Na zona ideal de combate
            if self.shot_this_tick and has_los:
                rewards[0] += 0.10  # Bônus por atirar com visão clara
        elif dist_to_enemy > 15.0:
            # Precisa se aproximar
            if dist_delta < -0.1:
                rewards[0] = 0.10  # Aproximando, bom
            else:
                rewards[0] = -0.10  # Deveria estar aproximando
        else:
            # Muito perto (< 8), deveria recuar para range ideal
            if dist_delta > 0.1:
                rewards[0] = 0.05  # Recuando para range
            else:
                rewards[0] = -0.05  # Muito perto e não recua
        
        # 1: EXECUTAR — All-in: perseguir implacavelmente + atirar sem parar
        if dist_delta < -0.1:
            rewards[1] = 0.20  # Fechando distância
        elif dist_delta > 0.1:
            rewards[1] = -0.15  # Recuando quando deveria executar
        else:
            rewards[1] = -0.05  # Parado
        if self.shot_this_tick:
            rewards[1] += 0.15  # Atirar é obrigatório no all-in
        else:
            rewards[1] -= 0.10  # Hesitação é punida
        if dist_to_enemy < 5.0 and self.shot_this_tick:
            rewards[1] += 0.20  # Bônus melee: atirar de perto
        
        # 2: DEFENDER — Manter distância > 15, buscar cobertura, atirar quando seguro
        if dist_to_enemy > 15.0:
            rewards[2] = 0.10  # Distância segura
            if not has_los:
                rewards[2] += 0.10  # Cobertura = excelente
        elif dist_to_enemy < 10.0:
            rewards[2] = -0.15  # Muito perto quando deveria defender
        else:
            rewards[2] = 0.0  # Zona cinza
        
        if dist_delta > 0.1:
            rewards[2] += 0.05  # Aumentando distância, bom para defesa
        
        # Atirar defensivamente: só vale se tem LOS e distância segura
        if self.shot_this_tick and has_los and dist_to_enemy > 12.0:
            rewards[2] += 0.10  # Tiro seguro de longe
        
        # 3: FUGIR — Maximizar distância, ignorar combate, buscar HP drops
        if dist_delta > 0.2:
            rewards[3] = 0.20  # Fugindo efetivamente
        elif dist_delta > 0.0:
            rewards[3] = 0.05  # Fugindo devagar
        elif dist_delta < -0.1:
            rewards[3] = -0.20  # Aproximando quando deveria fugir
        else:
            rewards[3] = -0.10  # Parado quando deveria fugir
        
        # Buscar HP drops durante fuga
        drop_positions = np.argwhere(self.map_drops == 1)
        if len(drop_positions) > 0:
            dists_to_drops = np.linalg.norm(drop_positions - self.player_pos, axis=1)
            closest_drop_dist = np.min(dists_to_drops)
            if closest_drop_dist < 5.0:
                rewards[3] += 0.15  # Perto de cura durante fuga
        
        # Penaliza atirar durante fuga (desperdício de tempo)
        if self.shot_this_tick:
            rewards[3] -= 0.05
        
        # 4: ENCURRALAR — Empurrar inimigo contra parede/canto
        if enemy_wall_dist < 8.0:
            rewards[4] = 0.15  # Inimigo perto da parede, bom
            if enemy_wall_dist < 4.0:
                rewards[4] = 0.25  # Inimigo no canto, excelente
        else:
            rewards[4] = -0.05  # Inimigo longe das paredes
        
        # Recompensa por empurrar inimigo em direção à parede
        if wall_delta < -0.1:
            rewards[4] += 0.10  # Inimigo se aproximando da parede
        elif wall_delta > 0.1:
            rewards[4] -= 0.10  # Inimigo escapando do canto
        
        # Manter pressão: atirar enquanto encurrala
        if self.shot_this_tick and enemy_wall_dist < 8.0:
            rewards[4] += 0.10
        
        # Evitar que o jogador se encurrale no processo
        if player_wall_dist < 3.0:
            rewards[4] -= 0.10  # Se encurralar é ruim
        
        # 5: CONTROLAR_MAPA — Posição central, patrulhar, coletar drops
        arena_half = (arena_max - arena_min) / 2.0
        if dist_to_center < arena_half * 0.3:
            rewards[5] = 0.10  # No centro, controle territorial
        elif dist_to_center < arena_half * 0.5:
            rewards[5] = 0.05  # Razoavelmente central
        else:
            rewards[5] = -0.05  # Nas bordas
        
        # Coletar drops quando em controle de mapa
        if len(drop_positions) > 0:
            if closest_drop_dist < 8.0:
                rewards[5] += 0.10
        
        # Atirar quando oportuno (sem exigir posição específica)
        if self.shot_this_tick and has_los:
            rewards[5] += 0.05
        
        # --- Atualiza estado anterior ---
        self.prev_dist_to_enemy = dist_to_enemy
        self.prev_enemy_wall_dist = enemy_wall_dist
        
        # Produto escalar: isola apenas a recompensa do estado ativo
        return float(np.dot(rewards, self.current_intent))

    def step(self, action, enemy_action=None, curriculum_progress=None):
        if curriculum_progress is not None:
            self.curriculum_progress = float(np.clip(curriculum_progress, 0.0, 1.0))
        
        p = self.curriculum_progress
        reward = 0.0
        done = False
        
        # Salva direção de movimento para intent reward
        if "move_idx" in (action if action else {}):
            move_idx = action["move_idx"]
            self.prev_move_dir = self.move_dirs[move_idx].copy()
        elif action and "move" in action:
            self.prev_move_dir = np.array(action["move"], dtype=np.float32)
        else:
            self.prev_move_dir = np.zeros(2, dtype=np.float32)
        
        # =============================================
        # FÍSICA INTERPOLADA (LERP) — calculada a cada tick
        # =============================================
        current_hitbox = self.lerp(self.hitbox_radius_max, self.hitbox_radius_min, p)
        current_enemy_speed = self.lerp(self.enemy_speed_min, self.enemy_speed_max, p)
        current_enemy_proj_speed = self.lerp(self.enemy_proj_speed_min, self.enemy_proj_speed_max, p)
        current_shoot_interval = int(self.lerp(self.enemy_shoot_interval_min, self.enemy_shoot_interval_max, p))
        
        # Gotejamento de Ameaça: probabilidade de ataque pesado
        prob_heavy_attack = max(0.0, (p - 0.5) * 0.1)
        
        # Arena bounds para este tick
        arena_min, arena_max = self.get_arena_bounds()
        
        # Skill mask para este tick
        skill_mask = self.get_skill_mask()
        
        # Bounce habilitado?
        bounce_enabled = p >= self.threshold_bounce
        
        # Atualização de status e cooldowns
        self.player_stats[1:] = np.maximum(0.0, self.player_stats[1:] - 1.0)
        self.enemy_stats[1:] = np.maximum(0.0, self.enemy_stats[1:] - 1.0)
        self.proj_cooldown = max(0, self.proj_cooldown - 1)
        self.enemy_proj_cooldown = max(0, self.enemy_proj_cooldown - 1)
        
        # --- SHAPING: Punição de Inércia, Tempo e Coleta ---
        reward += self.update_rewards()
        
        # --- RECOMPENSA TÁTICA (Intent-driven) ---
        intent_reward = self.compute_intent_reward() * self.intent_scale
        reward += intent_reward
        self.reward_from_intent += intent_reward
        self.shot_this_tick = False  # Reset flag a cada tick
        
        if self.enemy_stats[2] > 0 and self.enemy_stats[2] % 30 == 0:
            self.enemy_stats[0] -= 5
            reward += 1.0 
        
        # --- PLAYER MOVEMENT ---
        speed = 0.8 if self.player_stats[9] > 0 else self.player_speed_base
        
        if p < self.threshold_movement:
            # Fase torreta: movimento bloqueado
            move_vec = np.array([0.0, 0.0], dtype=np.float32)
        elif "move_idx" in action:
            move_idx = action["move_idx"]
            move_vec = self.move_dirs[move_idx] * speed
        else:
            move_vec = np.array(action.get("move", [0.0, 0.0]), dtype=np.float32)
            if np.linalg.norm(move_vec) > 0.0:
                move_vec = np.clip(move_vec, -1.0, 1.0) * speed
                
        noclip = self.player_stats[1] > 0
        if np.linalg.norm(move_vec) > 0.0:
            new_px = self.player_pos[0] + move_vec[0]
            # Colisão com parede E com arena
            if noclip or (self.is_inside_arena(new_px, self.player_pos[1]) and 
                          not self.map_walls[int(new_px), int(self.player_pos[1])]):
                self.player_pos[0] = np.clip(new_px, arena_min, arena_max - 1)
            new_py = self.player_pos[1] + move_vec[1]
            if noclip or (self.is_inside_arena(self.player_pos[0], new_py) and
                          not self.map_walls[int(self.player_pos[0]), int(new_py)]):
                self.player_pos[1] = np.clip(new_py, arena_min, arena_max - 1)
                
        # --- COMBAT & AIM ---
        aim_vec = np.array(action.get("aim", [1.0, 0.0]), dtype=np.float32)
        norm = np.linalg.norm(aim_vec)
        if norm > 0: aim_vec /= norm
        
        if self.last_aim_vec is not None:
            if np.linalg.norm(aim_vec - self.last_aim_vec) > 0.05:
                self.has_rotated_aim = True
        self.last_aim_vec = aim_vec.copy()
        
        # Tiros do Jogador (cada disparo custa -0.05 para punir spray-and-pray)
        if action.get("shoot") and self.proj_cooldown == 0:
            self.shots_fired += 1
            self.shot_this_tick = True
            reward -= 0.05  # Custo de munição
            self.reward_from_penalties -= 0.05
            inactive = np.where(~self.proj_active)[0]
            if len(inactive) > 0:
                idx = inactive[0]
                self.proj_active[idx] = True
                self.proj_pos[idx] = self.player_pos.copy()
                self.proj_vel[idx] = aim_vec * 1.5
                self.proj_bounces[idx] = 0
                ptype = 0
                if self.player_stats[7] > 0: ptype = 2
                elif self.player_stats[8] > 0: ptype = 3
                self.proj_type[idx] = ptype
                self.proj_owner[idx] = 0
                self.proj_cooldown = 3
                
        # Bônus de Mira — aplicado APENAS no tick em que o agente dispara
        if self.shot_this_tick:
            to_enemy = self.enemy_pos - self.player_pos
            dist = np.linalg.norm(to_enemy)
            if dist > 0:
                to_enemy_norm = to_enemy / dist
                cos_theta = np.dot(aim_vec, to_enemy_norm)
                aim_bonus = 0.1 * max(0.0, cos_theta)
                reward += aim_bonus  # Bônus proporcional à precisão do disparo
                self.reward_from_aim_bonus += aim_bonus
                
        # --- ATUALIZAÇÃO DE PROJÉTEIS ---
        if np.any(self.proj_active):
            active_mask = self.proj_active.copy() # Congela a máscara deste frame
            new_pos = self.proj_pos[active_mask] + self.proj_vel[active_mask]
            self.proj_pos[active_mask] = new_pos
            
            out_x = (new_pos[:, 0] <= 0) | (new_pos[:, 0] >= self.map_size - 1)
            out_y = (new_pos[:, 1] <= 0) | (new_pos[:, 1] >= self.map_size - 1)
            bounced_border = out_x | out_y
            
            if not bounce_enabled:
                # Sem ricochete: projéteis morrem na borda
                idx_border = np.where(active_mask)[0][bounced_border]
                player_misses = np.sum(self.proj_owner[idx_border] == 0)
                enemy_misses = np.sum(self.proj_owner[idx_border] == 1)
                if player_misses > 0:
                    self.consecutive_hits = 0
                    self.player_hit_streak = 0
                    miss_penalty = 0.3 * player_misses
                    reward -= miss_penalty
                    self.reward_from_penalties -= miss_penalty
                self.proj_active[idx_border] = False
                if enemy_misses > 0:
                    self.enemy_miss_streak += enemy_misses
                    if self.enemy_miss_streak >= 3:
                        try:
                            self.event_queue.put_nowait({"type": "enemy_miss_streak", "count": self.enemy_miss_streak})
                        except queue.Full:
                            pass
            else:
                # Ricochete habilitado
                self.proj_vel[active_mask, 0] = np.where(out_x, -self.proj_vel[active_mask, 0], self.proj_vel[active_mask, 0])
                self.proj_vel[active_mask, 1] = np.where(out_y, -self.proj_vel[active_mask, 1], self.proj_vel[active_mask, 1])
                self.proj_bounces[active_mask] += np.where(bounced_border, 1, 0)
            
            # Recalcula a máscara para colisões internas
            active = self.proj_active
            px_int = np.clip(self.proj_pos[active, 0].astype(int), 0, self.map_size-1)
            py_int = np.clip(self.proj_pos[active, 1].astype(int), 0, self.map_size-1)
            hit_wall = self.map_walls[px_int, py_int] == 1
            
            if np.any(hit_wall):
                idx_hit = np.where(active)[0][hit_wall]
                
                if not bounce_enabled:
                    player_wall_misses = np.sum(self.proj_owner[idx_hit] == 0)
                    if player_wall_misses > 0:
                        self.consecutive_hits = 0
                        wall_miss_penalty = 0.3 * player_wall_misses
                        reward -= wall_miss_penalty
                        self.reward_from_penalties -= wall_miss_penalty
                    self.proj_active[idx_hit] = False
                else:
                    # Ricochete realístico para colisões internas
                    prev_pos = self.proj_pos[idx_hit] - self.proj_vel[idx_hit]
                    prev_x = np.clip(prev_pos[:, 0].astype(int), 0, self.map_size-1)
                    prev_y = np.clip(prev_pos[:, 1].astype(int), 0, self.map_size-1)
                    curr_x = px_int[hit_wall]
                    curr_y = py_int[hit_wall]
                    
                    flip_x = self.map_walls[curr_x, prev_y] == 1
                    flip_y = self.map_walls[prev_x, curr_y] == 1
                    
                    # Quina perfeita: se nem X nem Y for parede (mas a diagonal for), inverte os dois
                    corner = (~flip_x) & (~flip_y)
                    flip_x = flip_x | corner
                    flip_y = flip_y | corner
                    
                    self.proj_vel[idx_hit, 0] = np.where(flip_x, -self.proj_vel[idx_hit, 0], self.proj_vel[idx_hit, 0])
                    self.proj_vel[idx_hit, 1] = np.where(flip_y, -self.proj_vel[idx_hit, 1], self.proj_vel[idx_hit, 1])
                    self.proj_bounces[idx_hit] += 1
                
            too_many = self.proj_bounces[active] > 3
            if np.any(too_many):
                idx_too_many = np.where(active)[0][too_many]
                player_bounce_misses = np.sum(self.proj_owner[idx_too_many] == 0)
                if player_bounce_misses > 0:
                    self.consecutive_hits = 0
                    bounce_miss_penalty = 0.3 * player_bounce_misses
                    reward -= bounce_miss_penalty
                    self.reward_from_penalties -= bounce_miss_penalty
                self.proj_active[idx_too_many] = False
                
            # Dano no Inimigo — hitbox via LERP contínuo
            dist_enemy = np.linalg.norm(self.proj_pos[active] - self.enemy_pos, axis=1)
            hit_enemy = (dist_enemy < current_hitbox) & (self.proj_owner[active] == 0)
            if np.any(hit_enemy):
                for j, i in enumerate(np.where(active)[0]):
                    if not hit_enemy[j]: continue
                    self.consecutive_hits += 1
                    self.shots_hit += 1
                    self.player_hit_streak += 1
                    self.enemy_miss_streak = 0
                    
                    # Event: Player combo
                    if self.player_hit_streak >= 2 and self.player_hit_streak % 2 == 0:
                        try:
                            self.event_queue.put_nowait({
                                "type": "player_combo",
                                "count": self.player_hit_streak,
                                "player_hp": self.player_stats[0],
                                "enemy_hp": self.enemy_stats[0],
                                "distance": np.linalg.norm(self.player_pos - self.enemy_pos)
                            })
                        except queue.Full:
                            pass
                    dist = dist_enemy[j]
                    pt = self.proj_type[i]
                    pb = self.proj_bounces[i]
                    
                    # Bônus de precisão ("Na mosca" = +1.0 extra, Borda = +0.0 extra)
                    precision_bonus = max(0.0, 1.0 - (dist / current_hitbox))
                    combo_bonus = min(10.0, 0.5 * self.consecutive_hits)
                    base_rew = 3.0 + precision_bonus + combo_bonus
                    
                    if pb > 0:
                        base_rew *= 0.5 # Ricochete recompensa menos
                        self.fatal_bounce = True
                    else:
                        self.fatal_bounce = False
                        
                    reward += base_rew
                    self.reward_from_hits += base_rew
                    
                    if pt == 1: self.enemy_stats[0] -= 50
                    elif pt == 2: self.enemy_stats[0] -= 10; self.enemy_stats[1] = 150
                    elif pt == 3: self.enemy_stats[0] -= 10; self.enemy_stats[2] = 150
                    else: self.enemy_stats[0] -= 10
                    
                    # Event: Enemy low HP
                    if self.enemy_stats[0] > 0 and self.enemy_stats[0] < 20:
                        try:
                            self.event_queue.put_nowait({
                                "type": "enemy_low_hp", 
                                "hp": float(self.enemy_stats[0]),
                                "player_hp": self.player_stats[0],
                                "distance": np.linalg.norm(self.player_pos - self.enemy_pos)
                            })
                        except queue.Full:
                            pass
                    
                self.proj_active[np.where(active)[0][hit_enemy]] = False
                
            # Dano no Player
            dist_player = np.linalg.norm(self.proj_pos[active] - self.player_pos, axis=1)
            hit_player = (dist_player < 1.0) & (self.proj_owner[active] == 1)
            if np.any(hit_player):
                idx_player_hit = np.where(active)[0][hit_player]
                if self.player_stats[1] > 0: # Shield / Noclip ativado
                    self.proj_vel[idx_player_hit] *= -1
                    self.proj_owner[idx_player_hit] = 0 
                else:
                    hits = np.sum(hit_player)
                    
                    if self.tick - self.last_hit_tick <= 30: # 1 segundo (30 ticks)
                        self.consecutive_hits += hits
                    else:
                        self.consecutive_hits = hits
                    self.last_hit_tick = self.tick
                    
                    base_penalty = hits * 5.0
                    combo_penalty = (self.consecutive_hits - 1) * 2.0 if self.consecutive_hits > 1 else 0.0
                    total_penalty = base_penalty + combo_penalty
                    
                    self.player_stats[0] -= hits * 10
                    reward -= total_penalty
                    self.reward_from_penalties -= total_penalty
                    self.proj_active[idx_player_hit] = False
                    
        # --- GERAÇÃO DINÂMICA DE ITENS (Dense Reward Signal) ---
        # Itens aparecem a partir de progress >= 0.15
        if p >= 0.15 and np.random.rand() < 0.05:
            if np.sum(self.map_drops == 1) < 5:
                a_min_i, a_max_i = int(self.get_arena_bounds()[0]), int(self.get_arena_bounds()[1])
                if a_max_i > a_min_i:
                    rx = np.random.randint(a_min_i, a_max_i)
                    ry = np.random.randint(a_min_i, a_max_i)
                    if not self.map_walls[rx, ry] and self.map_drops[rx, ry] == 0:
                        self.map_drops[rx, ry] = 1
            
        # =============================================
        # IA INIMIGA — Unificada por Probability Scaling
        # =============================================
        if enemy_action is not None:
            # Self-play ou ação externa
            emove_idx = enemy_action.get("move_idx", 0)
            emove_vec = self.move_dirs[emove_idx].copy()
            e_speed = current_enemy_speed if self.enemy_stats[1] == 0 else current_enemy_speed * 0.3
            if np.linalg.norm(emove_vec) > 0.0:
                emove_vec = np.clip(emove_vec, -1.0, 1.0) * e_speed
                new_ex = self.enemy_pos[0] + emove_vec[0]
                if self.is_inside_arena(new_ex, self.enemy_pos[1]) and not self.map_walls[int(new_ex), int(self.enemy_pos[1])]:
                    self.enemy_pos[0] = np.clip(new_ex, arena_min, arena_max - 1)
                new_ey = self.enemy_pos[1] + emove_vec[1]
                if self.is_inside_arena(self.enemy_pos[0], new_ey) and not self.map_walls[int(self.enemy_pos[0]), int(new_ey)]:
                    self.enemy_pos[1] = np.clip(new_ey, arena_min, arena_max - 1)
                    
            eaim_vec = np.array(enemy_action.get("aim", [1.0, 0.0]), dtype=np.float32)
            enorm = np.linalg.norm(eaim_vec)
            if enorm > 0: eaim_vec /= enorm
            
            if enemy_action.get("shoot") and self.enemy_proj_cooldown == 0:
                inactive = np.where(~self.proj_active)[0]
                if len(inactive) > 0:
                    idx = inactive[0]
                    self.proj_active[idx] = True
                    self.proj_pos[idx] = self.enemy_pos.copy()
                    self.proj_vel[idx] = eaim_vec * current_enemy_proj_speed
                    self.proj_bounces[idx] = 0
                    self.proj_type[idx] = 0
                    self.proj_owner[idx] = 1
                    self.enemy_proj_cooldown = int(current_shoot_interval)
        else:
            # IA Heurística Contínua (sem branches por nível)
            real_vec = self.player_pos - self.enemy_pos
            real_dist = np.linalg.norm(real_vec)
            if real_dist > 0:
                real_vec_norm = real_vec / real_dist
            else:
                real_vec_norm = np.array([0.0, 1.0], dtype=np.float32)
            
            # Movimento do Inimigo
            if p >= self.threshold_enemy_ai:
                e_speed_actual = current_enemy_speed
                if self.enemy_stats[1] > 0:
                    e_speed_actual *= 0.3
                    
                if p >= self.threshold_enemy_astar and real_dist > 5:
                    # A* pathfinding para inimigo avançado
                    next_pos = self.compute_pathfinding(self.enemy_pos, self.player_pos)
                    enemy_vec = next_pos - self.enemy_pos
                    enemy_dist = np.linalg.norm(enemy_vec)
                    if enemy_dist > 0:
                        enemy_vec /= enemy_dist
                    
                    new_ex = self.enemy_pos[0] + enemy_vec[0] * e_speed_actual
                    if self.is_inside_arena(new_ex, self.enemy_pos[1]) and not self.map_walls[int(new_ex), int(self.enemy_pos[1])]:
                        self.enemy_pos[0] = new_ex
                    new_ey = self.enemy_pos[1] + enemy_vec[1] * e_speed_actual
                    if self.is_inside_arena(self.enemy_pos[0], new_ey) and not self.map_walls[int(self.enemy_pos[0]), int(new_ey)]:
                        self.enemy_pos[1] = new_ey
                elif np.random.rand() < 0.2:
                    # Random walk simples
                    move_idx = np.random.randint(1, 5)
                    move_vec_e = self.move_dirs[move_idx] * e_speed_actual
                    new_ex = self.enemy_pos[0] + move_vec_e[0]
                    if self.is_inside_arena(new_ex, self.enemy_pos[1]) and not self.map_walls[int(new_ex), int(self.enemy_pos[1])]:
                        self.enemy_pos[0] = np.clip(new_ex, arena_min, arena_max - 1)
                    new_ey = self.enemy_pos[1] + move_vec_e[1]
                    if self.is_inside_arena(self.enemy_pos[0], new_ey) and not self.map_walls[int(self.enemy_pos[0]), int(new_ey)]:
                        self.enemy_pos[1] = np.clip(new_ey, arena_min, arena_max - 1)
            
            # Tiro do Inimigo — probabilidade escalada com progress
            if p >= self.threshold_enemy_shoot:
                # Probabilidade de tiro por tick baseada no intervalo interpolado
                shoot_prob = 1.0 / max(1, current_shoot_interval)
                if self.tick % max(1, current_shoot_interval) == 0 or np.random.rand() < shoot_prob * 0.1:
                    inactive = np.where(~self.proj_active)[0]
                    if len(inactive) > 0:
                        idx = inactive[0]
                        self.proj_active[idx] = True
                        self.proj_pos[idx] = self.enemy_pos.copy()
                        self.proj_vel[idx] = real_vec_norm * current_enemy_proj_speed
                        self.proj_bounces[idx] = 0
                        self.proj_type[idx] = 0
                        self.proj_owner[idx] = 1
            
            # Gotejamento de Ameaça: ataques pesados (forçam uso de Dash/Shield)
            if prob_heavy_attack > 0 and np.random.rand() < prob_heavy_attack:
                inactive = np.where(~self.proj_active)[0]
                # Dispara 3 projéteis em leque (ameaça de área)
                for spread in [-0.3, 0.0, 0.3]:
                    if len(inactive) > 0:
                        idx = inactive[0]
                        inactive = inactive[1:]
                        self.proj_active[idx] = True
                        self.proj_pos[idx] = self.enemy_pos.copy()
                        spread_vec = np.array([
                            real_vec_norm[0] * np.cos(spread) - real_vec_norm[1] * np.sin(spread),
                            real_vec_norm[0] * np.sin(spread) + real_vec_norm[1] * np.cos(spread)
                        ], dtype=np.float32)
                        self.proj_vel[idx] = spread_vec * current_enemy_proj_speed * 1.5
                        self.proj_bounces[idx] = 0
                        self.proj_type[idx] = 0
                        self.proj_owner[idx] = 1
                
        # --- CONDIÇÕES DE TÉRMINO ---
        if self.enemy_stats[0] <= 0:
            bonus = 50.0 / max(1.0, float(self.shots_fired))
            if self.has_rotated_aim:
                bonus += 5.0
                
            kill_reward = 15.0 if getattr(self, 'fatal_bounce', False) else 50.0
            
            kill_total = kill_reward + bonus
            reward += kill_total
            self.reward_from_kills += kill_total
            self.kill_tick = self.tick
            self.episode_result = 'kill'
            done = True
            try:
                self.event_queue.put_nowait({
                    "type": "player_kill", 
                    "tick": self.tick, 
                    "shots": self.shots_fired,
                    "player_hp": self.player_stats[0],
                    "enemy_hp": self.enemy_stats[0],
                    "distance": np.linalg.norm(self.player_pos - self.enemy_pos)
                })
            except queue.Full:
                pass
            
        elif self.player_stats[0] <= 0:
            reward -= 10.0
            self.reward_from_penalties -= 10.0
            self.episode_result = 'death'
            done = True
            try:
                self.event_queue.put_nowait({
                    "type": "player_death",
                    "tick": self.tick,
                    "reason": self.death_reason,
                    "player_hp": self.player_stats[0],
                    "enemy_hp": self.enemy_stats[0],
                    "distance": np.linalg.norm(self.player_pos - self.enemy_pos)
                })
            except queue.Full:
                pass
            
        # Escala o tempo limite proporcionalmente à vida (para dar tempo de se matarem com 1000 HP)
        hp_multiplier = self.player_hp_base / 100.0
        max_ticks = int(self.lerp(500, 1000, p) * hp_multiplier)

        if self.tick >= max_ticks:
            reward -= 50.0  # Penalidade de timeout extrema
            self.reward_from_penalties -= 50.0
            self.episode_result = 'timeout'
            done = True
            
        self.push_frame()
            
        return self.get_state(), reward, done, {}
