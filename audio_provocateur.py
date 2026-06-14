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
        """Retorna um taunt hardcoded aleatório de alta provocação caso a geração falhe."""
        event_type = event.get("type")
        
        taunts = {
            "enemy_miss_streak": [
                "Você é cego ou apenas incompetente? Sua magia é fraca!",
                "Você atira no ar enquanto eu manipulo as strings da realidade!",
                "Sua mira é tão patética quanto a sua linhagem, humano!",
                "Desista! Você não passa de um inseto cego perante um Mago Supremo!",
                "Estou calculando os vetores da sua incompetência. É fascinante.",
                "O ar está sangrando, mas minha barreira de mana continua intacta, verme.",
                "Golpear o vazio é a base da sua escola de combate ou apenas dano cerebral?",
                "Minha presença passiva exige mais destreza do que você jamais terá em toda a vida.",
                "Você estuda espadas e flechas enquanto eu reescrevo as leis da termodinâmica local.",
                "Nem um constructo defeituoso de argila erraria ataques com tamanha precisão estatística."
            ],
            "player_combo": [
                "Acha que essa faísca me machuca? Eu sou a própria tempestade!",
                "Seus golpes são apenas cócegas na armadura do arcano!",
                "Desfrute do seu momento de glória, mortal. Ele será o último!",
                "Toda essa coreografia patética para arranhar meus escudos ilusórios?",
                "Batendo contra mana cristalizado com força bruta? A ignorância é comovente.",
                "Aproveite o acúmulo de ácido lático; minha retaliação vai evaporar seus ossos.",
                "Cada acerto seu apenas serve como catalisador para a maldição que já está no seu sangue.",
                "Isso é o melhor do seu arsenal? Estou recebendo mais dano por tédio do que por seus ataques.",
                "Sua persistência beira a demência. Você está lutando contra a entropia pura!"
            ],
            "enemy_low_hp": [
                "Você acha que eu sangro? O que escorre de mim é plasma dimensional!",
                "A dor física apenas remove a trava de segurança do meu poder. Trema!",
                "Eu já retornei das cinzas centenas de vezes. Você, mortal, apodrece apenas uma vez!",
                "A casca cede, mas meu núcleo astral está prestes a entrar em colapso crítico!",
                "Você não me feriu; você apenas quebrou os selos que limitavam minha destruição em área!",
                "Até colapsando em poeira cósmica, eu conjuro cataclismos que sua mente símea não compreende!",
                "Meu sangue fervente queima os alicerces do seu mundo de vermes!",
                "O abismo me exige de volta, mas eu vou arrastar seu cadáver comigo para as profundezas!"
            ],
            "player_kill": [
                "Impossível! Um inseto não pode ferir uma anomalia cósmica!",
                "Isso não é o fim... Minha consciência já migrou para a rede de cristal da torre!",
                "Maldito primata! Aproveite os segundos residuais da sua sorte matemática!",
                "Sua vitória é um bug na matriz do destino. Eu serei o patch de execução brutal!",
                "Destruiu minha projeção astral? Irrelevante. Nos veremos no plano etéreo!",
                "Você não mata energia arcana, seu troglodita, você apenas sofre as consequências da dissipação!",
                "A carne apodrece, mas a frequência da minha alma sobrevive. Sua linhagem está marcada!",
                "Comemore na lama e na sua própria ignorância. Eu ascenderei às estrelas!"
            ],
            "player_death": [
                f"Sua alma agora me pertence! {event.get('reason', 'Destruído por atrito cósmico')}!",
                f"Você foi apagado da existência, reduzido a átomos inertes. {event.get('reason', 'Falha estrutural')}.",
                f"Tão frágil... Sua poeira vai adubar a botânica dos meus jardins venenosos. {event.get('reason', 'Desintegrado')}!",
                f"Você achou que o fim seria rápido? A relatividade distorcerá seus gritos por éons! {event.get('reason', 'Roto no espaço-tempo')}.",
                f"Levante-se! Eu quero dissecar seu espírito de novo! Ah, sim, mortalidade linear. {event.get('reason', 'Patético')}.",
                f"Isso foi deprimente. Eu calibrei feitiços de nono círculo para um saco de carne. {event.get('reason', 'Sobrecarga de mana')}.",
                f"Nem mesmo as leis da conservação de massa vão se importar com você. {event.get('reason', 'Anulado')}.",
                f"Manipulação da realidade não é brincadeira para primatas semi-evoluídos. {event.get('reason', 'Combustão espontânea')}!",
                f"Seu processamento neural parou antes mesmo da minha magia te atingir. {event.get('reason', 'Morte por choque arcano')}."
            ],
            "player_idle": [
                "Congelou de pavor existencial? É a reação neurológica correta perante um lorde!",
                "Aceite a morte inerte. Simplifica os cálculos da minha detonação.",
                "Seu cérebro atrofiou? O rigor mortis chegou antes do óbito?",
                "Ande logo, verme! Eu não tenho decaimento radioativo de milênios a perder com você!",
                "Meu feitiço de dilação temporal sequer foi ativado e você já é uma estátua geológica.",
                "O fluxo de mana não espera sua letargia. Defenda-se ou torne-se adubo orgânico instantâneo!",
                "Você respira por costume ou porque esqueceu como se para?",
                "A pedra sob suas botas exibe mais agência e inteligência em combate do que você."
            ],
            "player_fleeing": [
                "Aceleração linear não te salva de geometria não-euclidiana, tolo!",
                "Correr é inútil! Meus rastreadores de plasma mapeiam o medo no seu sistema nervoso!",
                "Pode correr, verme! Cinética básica não supera o teletransporte quântico!",
                "Dar as costas a um lorde da magia é encurtar o tempo de vida para a próxima fração de segundo!",
                "Vai dar as costas para a entropia? Eu distorço o espaço até seus calcanhares!",
                "Suas perninhas tremem sob a gravidade do seu próprio terror cósmico?",
                "Fugir não reverte o decaimento celular que minhas maldições já iniciaram!",
                "Sobrevivência não consta nas probabilidades que calculei para esta câmara!"
            ],
            "player_cornered": [
                "Preso contra as fundações como o rato insolente que você é!",
                "A topologia do seu desespero agora é um sistema fechado. Saboreie a asfixia!",
                "O labirinto de mana se estabilizou. Não há vetores de fuga para a sua massa corporal!",
                "Fim da linha, macaco pelado. As paredes e eu concordamos que sua existência é redundante.",
                "Encurralado. Hora de testar a resistência térmica do seu esqueleto ao fogo estelar.",
                "Para onde agora? Você não possui eixos extras para escapar dessa dimensão!",
                "Sinta a gravidade se fechar sobre você. O espaço tridimensional se tornou seu caixão!",
                "Seus limites geométricos foram traçados em sangue. Prepare-se para a extração da alma."
            ]
        }
        
        choices = taunts.get(event_type, ["O algoritmo da sua dor já está compilado!", "Falha anatômica iminente!", "Sinta o colapso absoluto!"])
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
