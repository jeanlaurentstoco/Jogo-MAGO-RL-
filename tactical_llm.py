"""
Módulo de Meta-Controlador Tático via LLM (Gemini).

Recebe o estado do jogo e retorna um vetor one-hot de intenção tática.
Usado SOMENTE durante gameplay (renderer.py), NUNCA durante treinamento.

Configuração:
    1. Copie .env.example para .env
    2. Preencha GEMINI_API_KEY com sua chave do Google AI Studio
       (https://aistudio.google.com/apikey)
"""

import numpy as np
import os
import time
import threading

# Lazy imports para não quebrar treinamento
def _load_dotenv():
    """Carrega variáveis do .env se disponível."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

INTENT_NAMES = [
    "AGRESSIVO_PERSEGUIR",     # 0
    "FLANQUEAR",               # 1
    "FORCAR_CANTO",            # 2
    "ALL_IN_LETAL",            # 3
    "EVASIVO_RECUAR",          # 4
    "BUSCAR_COBERTURA",        # 5
    "MOVIMENTO_ERRATICO",      # 6
    "SOBREVIVENCIA_TARTARUGA", # 7
    "CONTROLE_TERRITORIAL",    # 8
    "PATRULHA_PERIMETRAL",     # 9
    "CAZAR_RECURSOS",          # 10
]

INTENT_MAP = {name: i for i, name in enumerate(INTENT_NAMES)}

SYSTEM_PROMPT = """Você é o Tático-Mestre de uma arena de combate mágico. 
Você analisa o estado da batalha e decide a ÚNICA estratégia tática que o mago agente deve seguir AGORA.

Você DEVE responder com EXATAMENTE um dos seguintes IDs, sem explicação:

AGRESSIVO_PERSEGUIR - Quando o inimigo está fraco ou fugindo
FLANQUEAR - Quando o inimigo está parado mirando fixo
FORCAR_CANTO - Quando o inimigo está perto das bordas da arena
ALL_IN_LETAL - Quando o inimigo está a um tiro de morrer
EVASIVO_RECUAR - Quando o agente está com HP crítico
BUSCAR_COBERTURA - Quando há muitos projéteis ativos do inimigo
MOVIMENTO_ERRATICO - Quando o inimigo está acertando muito
SOBREVIVENCIA_TARTARUGA - Quando o agente está com HP vantajoso e o tempo está acabando
CONTROLE_TERRITORIAL - Situação neutra, manter posição central
PATRULHA_PERIMETRAL - Quando o agente tem vantagem de alcance
CAZAR_RECURSOS - Quando há itens de cura próximos

Responda APENAS com o ID. Nada mais."""


class TacticalLLM:
    """Meta-controlador tático que usa Gemini para decidir intenções."""
    
    def __init__(self, tick_interval=30):
        """
        Args:
            tick_interval: A cada quantos ticks chamar o LLM (default: 30 ≈ 1s)
        """
        _load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.tick_interval = tick_interval
        self.last_intent = np.zeros(len(INTENT_NAMES), dtype=np.float32)
        self.last_intent[8] = 1.0  # Default: CONTROLE_TERRITORIAL
        self.last_call_tick = -tick_interval  # Força primeira chamada
        self.model = None
        self._lock = threading.Lock()
        self._pending_intent = None
        self.rate_limited_until = 0  # Timestamp de quando a cota estará liberada
        
        if not self.api_key or self.api_key == "sua_chave_gemini_aqui":
            print("[TacticalLLM] AVISO: GEMINI_API_KEY não configurada. Usando fallback heurístico.")
            self.enabled = False
        else:
            try:
                import google.genai as genai
                self.client = genai.Client(api_key=self.api_key)
                self.model_name = "gemini-2.0-flash"
                self.enabled = True
                print(f"[TacticalLLM] Conectado ao Gemini ({self.model_name})")
            except ImportError:
                print("[TacticalLLM] AVISO: google-genai não instalado. pip install google-genai")
                self.enabled = False
            except Exception as e:
                print(f"[TacticalLLM] AVISO: Erro ao conectar: {e}")
                self.enabled = False
    
    def _build_prompt(self, game_state):
        """Constrói o prompt com o estado atual do jogo."""
        p_hp = game_state["player_stats"][0]
        e_hp = game_state["enemy_stats"][0]
        p_pos = game_state["player_pos"]
        e_pos = game_state["enemy_pos"]
        dist = np.linalg.norm(p_pos - e_pos)
        
        proj_active = game_state.get("proj_active", np.zeros(1))
        proj_owner = game_state.get("proj_owner", np.zeros(1))
        enemy_projs = np.sum(proj_active & (proj_owner == 1)) if len(proj_active) > 0 else 0
        
        n_drops = np.sum(game_state.get("drops", np.zeros((1, 1))) == 1)
        progress = game_state.get("curriculum_progress", 1.0)
        
        return (
            f"Estado da Batalha:\n"
            f"- HP Agente: {p_hp:.0f}/100\n"
            f"- HP Inimigo: {e_hp:.0f}\n"
            f"- Distância: {dist:.1f} blocos\n"
            f"- Projéteis inimigos ativos: {enemy_projs}\n"
            f"- Itens no chão: {n_drops}\n"
            f"- Progresso do currículo: {progress:.2f}\n"
            f"\nQual estratégia o agente deve usar AGORA?"
        )
    
    def _parse_response(self, text):
        """Extrai o ID de intenção da resposta do LLM."""
        text = text.strip().upper()
        for name in INTENT_NAMES:
            if name in text:
                return INTENT_MAP[name]
        return 8  # Fallback: CONTROLE_TERRITORIAL
    
    def _to_one_hot(self, idx):
        """Converte índice em vetor one-hot."""
        intent = np.zeros(len(INTENT_NAMES), dtype=np.float32)
        intent[idx] = 1.0
        return intent
    
    def _async_call(self, game_state):
        """Chamada assíncrona ao LLM em thread separada."""
        try:
            prompt = self._build_prompt(game_state)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}
                ]
            )
            text = response.text
            idx = self._parse_response(text)
            with self._lock:
                self._pending_intent = self._to_one_hot(idx)
        except Exception as e:
            err_msg = str(e)
            print(f"[TacticalLLM] Erro na chamada: {err_msg}")
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                self.rate_limited_until = time.time() + 60.0  # Bloqueia chamadas LLM por 60s
    
    def get_intent(self, game_state, current_tick):
        """Retorna o vetor de intenção tática atual.
        
        Chama o LLM assincronamente a cada tick_interval ticks.
        Entre chamadas, retorna o último intent calculado.
        
        Args:
            game_state: Dict retornado por engine.get_state()
            current_tick: Tick atual do jogo
        Returns:
            Vetor one-hot numpy (11,)
        """
        # Se estivermos rate limited ou sem API, usa a heurística síncrona
        if not self.enabled or time.time() < self.rate_limited_until:
            intent = np.zeros(len(INTENT_NAMES), dtype=np.float32)
            p_hp = game_state["player_stats"][0]
            e_hp = game_state["enemy_stats"][0]
            dist = np.linalg.norm(game_state["player_pos"] - game_state["enemy_pos"])
            
            if e_hp <= 10: intent[3] = 1.0
            elif p_hp < 30: intent[4] = 1.0
            elif e_hp < 50 and dist > 15: intent[0] = 1.0
            elif dist < 8: intent[6] = 1.0
            elif dist > 30: intent[0] = 1.0
            else: intent[8] = 1.0
            self.last_intent = intent
            return self.last_intent.copy()
            
        # Coleta resultado pendente de chamada anterior
        with self._lock:
            if self._pending_intent is not None:
                self.last_intent = self._pending_intent
                self._pending_intent = None
        
        # Dispara nova chamada se intervalo suficiente
        if (current_tick - self.last_call_tick) >= self.tick_interval:
            self.last_call_tick = current_tick
            thread = threading.Thread(target=self._async_call, args=(game_state,), daemon=True)
            thread.start()
        
        return self.last_intent.copy()
    
    def get_intent_name(self):
        """Retorna o nome da intenção ativa atual."""
        idx = np.argmax(self.last_intent)
        return INTENT_NAMES[idx]
