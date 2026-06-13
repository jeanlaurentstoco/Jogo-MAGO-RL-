"""
Pipeline de Áudio Assíncrono (Provocador Sarcástico).

Consome eventos da Event Queue da GameEngine e gera provocações em áudio
usando Gemini (texto) + ElevenLabs (TTS).

Configuração:
    1. Copie .env.example para .env
    2. Preencha GEMINI_API_KEY e ELEVENLABS_API_KEY
    3. Opcionalmente configure ELEVENLABS_VOICE_ID

Instalação:
    pip install google-genai gtts pygame
"""

import os
import time
import random
import threading
import tempfile
import queue as queue_module

# Lazy imports
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

# Prompts de provocação por tipo de evento
# IMPORTANTE: A persona é o BOT (Inimigo) provocando o HUMAN (Player)
TAUNT_PROMPTS = {
    "enemy_miss_streak": "Você (o mago boss) esquivou de {count} tiros seguidos do jogador humano. Provoque a mira ruim dele em no máximo 15 palavras em português, de forma sarcástica.",
    "player_combo": "O jogador humano te acertou {count} vezes seguidas. Reclame de forma arrogante e irritada em no máximo 15 palavras em português, dizendo que foi sorte.",
    "enemy_low_hp": "Você (o mago boss) está quase morto com {hp:.0f} HP, mas continua arrogante. Provoque o jogador dizendo que ele não tem a força final necessária em no máximo 15 palavras em português.",
    "player_kill": "O jogador te matou com {shots} tiros. Comente a sua própria derrota de forma sarcástica e ameaçadora em no máximo 15 palavras em português.",
    "player_death": "Você acaba de aniquilar o jogador humano. O motivo da morte dele foi: {reason}. Comemore sua vitória magistral antes de começar a próxima partida, provocando como ele morreu, em no máximo 20 palavras em português.",
    "player_idle": "O jogador humano está parado inerte. Provoque a falta de ação dele em no máximo 15 palavras em português.",
    "player_fleeing": "O jogador humano está correndo para longe de você (fugindo). Chame-o de covarde de forma sarcástica em no máximo 15 palavras em português.",
    "player_cornered": "Você encurralou o jogador humano contra a parede. Provoque-o dizendo que não há escapatória em no máximo 15 palavras em português.",
}


class AudioProvocateur(threading.Thread):
    """Worker assíncrono que consome eventos e gera provocações em áudio.
    
    Roda como daemon thread — morre automaticamente quando o programa principal termina.
    """
    
    def __init__(self, event_queue, cooldown_seconds=15):
        """
        Args:
            event_queue: queue.Queue da GameEngine
            cooldown_seconds: Tempo mínimo entre áudios (evita encavalamento)
        """
        super().__init__(daemon=True)
        _load_dotenv()
        
        self.event_queue = event_queue
        self.cooldown = cooldown_seconds
        self.last_play_time = 0
        self._stop_event = threading.Event()
        
        # Gemini (gerador de texto sarcástico)
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_client = None
        self.gemini_model = "gemini-2.0-flash"
        
        # ElevenLabs (TTS)
        self.elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        self.elevenlabs_client = None
        
        self.enabled = True
        self._init_clients()
    
    def _init_clients(self):
        """Inicializa configurações locais."""
        # Apenas gTTS será utilizado (100% local e gratuito).
        pass
        
        # Removermos ElevenLabs por completo
        # Usaremos gTTS nativo
        self.elevenlabs_client = None
        pass
    
    def _get_hardcoded_taunt(self, event):
        """Retorna um taunt hardcoded aleatório caso o Gemini esteja indisponível."""
        event_type = event.get("type")
        
        taunts = {
            "enemy_miss_streak": [
                "Você é cego ou apenas incompetente? Sua magia é fraca!",
                "Você atira no ar enquanto eu leio o tecido da realidade!",
                "Sua mira é tão patética quanto a sua linhagem, humano!",
                "Desista! Você não passa de um inseto cego perante um Mago Supremo!"
            ],
            "player_combo": [
                "Acha que essa faísca me machuca? Eu sou a própria tempestade!",
                "Seus golpes são apenas cócegas na armadura do arcano!",
                "Desfrute do seu momento de glória, mortal. Ele será o último!"
            ],
            "enemy_low_hp": [
                "Você acha que eu sangro? O que escorre de mim é pura magia!",
                "A dor apenas alimenta o meu poder! Trema perante mim!",
                "Eu já retornei das cinzas centenas de vezes. Você morre apenas uma vez!"
            ],
            "player_kill": [
                "Impossível! Um inseto não pode ferir um Deus Arcano!",
                "Isso não é o fim... Minha alma retornará do próprio abismo!",
                "Maldito humano! Aproveite sua sorte provisória!"
            ],
            "player_death": [
                f"Sua alma agora me pertence! Destruído {event.get('reason', 'tentando me enfrentar')}!",
                f"Você foi apagado da existência, como deveria ser. {event.get('reason', 'Morto em combate')}.",
                f"Tão frágil... Sua poeira vai adubar meus jardins mágicos. {event.get('reason', 'Destruído')}!",
                f"Você achou que a morte seria rápida? Seus gritos me divertem! {event.get('reason', 'Caído')}.",
                f"Levante-se! Eu quero matar você de novo! Ah, esqueci, humanos só morrem uma vez. {event.get('reason', 'Patético')}.",
                f"Isso foi... decepcionante. Eu esperava mais do que isso. {event.get('reason', 'Fim da linha')}.",
                f"Nem mesmo o inferno vai aceitar uma alma tão insignificante quanto a sua. {event.get('reason', 'Adeus')}.",
                f"Feitiçaria não é brincadeira para crianças! {event.get('reason', 'Queimado em cinzas')}."
            ],
            "player_idle": [
                "Congelou de pavor? É a reação correta perante um lorde!",
                "Aceite a morte parado. Pelo menos você morrerá de pé!",
                "Seu cérebro parou de funcionar? Minha magia fará o resto!",
                "Ande logo, verme! Eu não tenho milênios a perder com você!"
            ],
            "player_fleeing": [
                "Fugir não vai curar suas feridas. O fogo arcano vai te encontrar!",
                "Correr é inútil! Meus feitiços rasgam a trama do espaço!",
                "Pode correr, verme! Eu amo caçar coisas fracas como você!",
                "Dar as costas a um lorde da magia é implorar pela morte!",
                "Vire-se e queime com honra, sua praga covarde!",
                "Suas perninhas tremem de pavor cósmico?"
            ],
            "player_cornered": [
                "Preso contra as pedras como o verme que você é!",
                "Sinta o desespero! Não há como fugir da minha fúria arcanista!",
                "O labirinto se fechou, e o Minotauro mágico chegou para jantar!"
            ]
        }
        
        choices = taunts.get(event_type, ["Prepare-se para sofrer!", "Você é fraco!", "Sinta meu poder!"])
        return random.choice(choices)

    def _generate_taunt(self, event):
        """Retorna uma provocação a partir das frases curadas locais."""
        if not self.enabled:
            return None
        return self._get_hardcoded_taunt(event)
    
    def _speak(self, text):
        """Converte texto em áudio e reproduz via pygame mixer."""
        try:
            from gtts import gTTS
            
            # Gera áudio via gTTS (100% gratuito)
            tts = gTTS(text=text, lang='pt', slow=False)
            
            # Salva em arquivo temporário
            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_file.close()
            tts.save(tmp_file.name)
            
            # Reproduz via pygame (non-blocking)
            try:
                import pygame.mixer
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                # Espera terminar para deletar o arquivo
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"[AudioProvocateur] Erro reproduzindo áudio: {e}")
            finally:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass
                    
        except Exception as e:
            print(f"[AudioProvocateur] Erro no TTS: {e}")
    
    def stop(self):
        """Sinaliza para o worker parar."""
        self._stop_event.set()
    
    def run(self):
        """Loop principal do worker. Consome eventos da fila."""
        if not self.enabled:
            return
        
        print("[AudioProvocateur] Worker de áudio iniciado. Aguardando eventos...")
        
        while not self._stop_event.is_set():
            try:
                # Bloqueia por até 1 segundo esperando evento
                event = self.event_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue
            
            # Cooldown check
            now = time.time()
            if now - self.last_play_time < self.cooldown:
                continue
            
            # Marca o tempo AGORA para evitar spam em caso de erro da API
            self.last_play_time = now
            
            # Gera provocação
            taunt_text = self._generate_taunt(event)
            if taunt_text:
                print(f"[AudioProvocateur] 🎙️ \"{taunt_text}\"")
                self._speak(taunt_text)
