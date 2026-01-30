import argparse
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tkinter as tk
except ImportError as exc:
    raise SystemExit("tkinter nao esta disponivel nesta instalacao do Python.") from exc


# ------------------------------
# Configuracoes do jogo
# ------------------------------
WIDTH = 420
HEIGHT = 640
FLOOR_HEIGHT = 80
FPS = 60
FIXED_DT = 1 / 120

BIRD_RADIUS = 16
GRAVITY = 2000.0
JUMP_VELOCITY = -520.0
TERMINAL_VELOCITY = 900.0

PIPE_WIDTH = 70
PIPE_GAP = 165
PIPE_SPEED = 220.0
PIPE_INTERVAL = 1.25
PIPE_MARGIN = 20

BG_COLOR = "#cfe9ff"
PIPE_COLOR = "#2bb673"
FLOOR_COLOR = "#d1a46a"
TEXT_COLOR = "#1e1e1e"

BIRD_X = WIDTH * 0.28
PLAY_HEIGHT = HEIGHT - FLOOR_HEIGHT

# ------------------------------
# Configuracoes de IA melhoradas
# ------------------------------
AI_QTABLE_PATH = "ai_qtable_enhanced.pkl"

AI_DX_BIN = 20
AI_DY_BIN = 20
AI_VY_BIN = 40

AI_ALPHA = 0.2
AI_GAMMA = 0.97
AI_EPS_START = 0.6
AI_EPS_MIN = 0.05
AI_EPS_DECAY = 0.985

# === MELHORIA 1: Sistema de recompensas aprimorado ===
AI_STEP_REWARD = 0.05  # Aumentado: sobreviver é bom
AI_PIPE_REWARD = 5.0   # MUITO aumentado: passar cano é ótimo!
AI_CENTER_REWARD = 0.3  # Aumentado: ficar no centro é importante
AI_FLAP_PENALTY = -0.02  # Penalidade leve
AI_DEAD_PENALTY = -10.0  # Penalidade pesada para morte
AI_SPEED_REWARD = 0.03  # NOVO: recompensa por velocidade controlada
AI_PREP_REWARD = 0.2    # NOVO: recompensa por preparação antes do cano

AI_DECISION_DT = 0.075
AI_RANDOM_FLAP_PROB = 0.10

# === MELHORIA 2: Configurações DQN aprimoradas ===
NN_MODEL_PATH = "ai_dqn_enhanced.pkl"
NN_DECISION_DT = 0.05

# Arquiteturas melhoradas
NN_LAYERS = (37, 256, 256, 128, 2)  # AUMENTADO: mais features de entrada (37 ao invés de 25)
NN_LAYERS_HEAVY = (37, 512, 512, 256, 128, 2)  # Ainda mais profunda
NN_LAYERS_DUELING = (37, 256, 256, 128, 2)  # Para Dueling DQN

NN_LR = 0.0008
NN_GAMMA = 0.99
NN_EPS_START = 1.0
NN_EPS_MIN = 0.02
NN_EPS_DECAY = 0.995
NN_EPS_MIN_SUCCESS = 0.0
NN_EPS_SUCCESS_DECAY = 0.90
NN_GUIDE_PROB = 0.9
NN_EPS_BOOST = 0.35
NN_EPS_MAX = 0.9
NN_FAIL_EPS_UP = 0.05
NN_EPS_DECAY_FAIL = 0.97
NN_EPS_REHEAT = 0.08
NN_WARMUP_EPISODES = 20
NN_CURRICULUM_EPISODES = 200
NN_ASSIST_NOISE = 60
NN_FAIL_GUIDE_TRIGGER = 8
NN_FAIL_GUIDE_EPISODES = 6
NN_TRAIN_STEPS = 1
NN_TRAIN_STEPS_HEAVY = 3
NN_TRAIN_STEPS_CUDA = 6

# === MELHORIA 3: Prioritized Experience Replay ===
NN_REPLAY_SIZE = 60000
NN_REPLAY_ALPHA = 0.6  # NOVO: Priorização
NN_REPLAY_BETA_START = 0.4  # NOVO: Importance sampling
NN_REPLAY_BETA_FRAMES = 100000  # NOVO: Annealing

NN_BATCH_SIZE = 64
NN_TRAIN_START = 1500
NN_TARGET_UPDATE = 800
NN_MAX_GRAD_NORM = 5.0

# === MELHORIA 4: Curriculum Learning Adaptativo ===
CURRICULUM_EASY_GAP = 200
CURRICULUM_NORMAL_GAP = 165
CURRICULUM_HARD_GAP = 140
CURRICULUM_EASY_SPEED = 200
CURRICULUM_NORMAL_SPEED = 220
CURRICULUM_HARD_SPEED = 250

TRAIN_SPEED_DEFAULT = 1.0
TRAIN_SPEED_MAX = 12.0

GRAPH_WIDTH = 220
GRAPH_HEIGHT = 150
GRAPH_PADDING = 10

STABLE_SCORE_ON = 6
STABLE_SCORE_OFF = 4
STABLE_MIN_HISTORY = 6

# === MELHORIA 5: Configurações de histórico ===
HISTORY_BUFFER_SIZE = 5  # NOVO: Manter histórico de frames


# ------------------------------
# Utilitarios
# ------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def require_numpy() -> None:
    if np is None:
        raise SystemExit("numpy nao esta instalado. Rode: pip install numpy")


def require_torch() -> None:
    if torch is None:
        raise SystemExit("torch nao esta instalado. Rode: pip install torch")


def get_torch_device(prefer_cuda: bool = True):
    require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_nn(heavy: bool, use_cuda: bool = False) -> None:
    global NN_LAYERS, NN_REPLAY_SIZE, NN_BATCH_SIZE, NN_TRAIN_START, NN_TARGET_UPDATE, NN_LR, NN_TRAIN_STEPS
    if heavy:
        NN_LAYERS = NN_LAYERS_HEAVY
        NN_REPLAY_SIZE = 300000 if use_cuda else 200000
        NN_BATCH_SIZE = 512 if use_cuda else 128
        NN_TRAIN_START = 8000 if use_cuda else 4000
        NN_TARGET_UPDATE = 1500 if use_cuda else 1200
        NN_LR = 0.0005 if use_cuda else 0.0006
        NN_TRAIN_STEPS = NN_TRAIN_STEPS_CUDA if use_cuda else NN_TRAIN_STEPS_HEAVY


def norm_signed(value: float, max_abs: float) -> float:
    if max_abs <= 0:
        return 0.0
    return clamp(value / max_abs, -1.0, 1.0)


def norm_unit(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return clamp(value / max_value, 0.0, 1.0)


def norm_unit_to_signed(value: float, max_value: float) -> float:
    return norm_unit(value, max_value) * 2.0 - 1.0


def predict_y(bird_y: float, bird_vy: float, t: float) -> float:
    return bird_y + bird_vy * t + 0.5 * GRAVITY * t * t


def assist_prob_for_episode(ep: int) -> float:
    if ep <= 0:
        return 1.0
    return clamp(1.0 - (ep / max(1, NN_CURRICULUM_EPISODES)), 0.0, 1.0)


# === MELHORIA 6: Curriculum adaptativo ===
def adaptive_difficulty(avg_score: float) -> Tuple[float, float]:
    """Ajusta dificuldade baseado no desempenho"""
    if avg_score > 50:
        # IA está dominando: dificultar
        return CURRICULUM_HARD_GAP, CURRICULUM_HARD_SPEED
    elif avg_score < 10:
        # IA está sofrendo: facilitar
        return CURRICULUM_EASY_GAP, CURRICULUM_EASY_SPEED
    else:
        # Dificuldade normal
        return CURRICULUM_NORMAL_GAP, CURRICULUM_NORMAL_SPEED


def sample_gap_y(bird_y: float, assist_prob: float) -> int:
    min_y = PIPE_GAP // 2 + PIPE_MARGIN
    max_y = HEIGHT - FLOOR_HEIGHT - PIPE_GAP // 2 - PIPE_MARGIN
    if random.random() < assist_prob:
        return int(clamp(bird_y + random.randint(-NN_ASSIST_NOISE, NN_ASSIST_NOISE), min_y, max_y))
    return random.randint(min_y, max_y)


def circle_rect_collision(cx: float, cy: float, r: float,
                           rx: float, ry: float, rw: float, rh: float) -> bool:
    nearest_x = clamp(cx, rx, rx + rw)
    nearest_y = clamp(cy, ry, ry + rh)
    dx = cx - nearest_x
    dy = cy - nearest_y
    return (dx * dx + dy * dy) <= (r * r)


def discretize(value: float, bin_size: int, min_value: float, max_value: float) -> int:
    v = clamp(value, min_value, max_value)
    bins = int((max_value - min_value) / bin_size)
    if bins <= 0:
        return 0
    idx = int((v - min_value) / bin_size)
    if idx >= bins:
        idx = bins - 1
    return idx


def make_state(dx: float, dy: float, vy: float) -> Tuple[int, int, int]:
    dx_i = discretize(dx, AI_DX_BIN, 0.0, WIDTH)
    dy_i = discretize(dy, AI_DY_BIN, -HEIGHT, HEIGHT)
    vy_i = discretize(vy, AI_VY_BIN, -TERMINAL_VELOCITY, TERMINAL_VELOCITY)
    return (dx_i, dy_i, vy_i)


def get_next_pipe(pipes, bird_x: float):
    next_pipe = None
    best_dx = 1e9
    for pipe in pipes:
        dx = pipe.x - bird_x
        if dx + PIPE_WIDTH >= 0 and dx < best_dx:
            best_dx = dx
            next_pipe = pipe
    return next_pipe


def state_values_from_pipes(bird_x: float, bird_y: float, bird_vy: float, pipes) -> Tuple[float, float, float]:
    next_pipe = get_next_pipe(pipes, bird_x)
    if next_pipe is None:
        dx = WIDTH
        dy = 0.0
    else:
        dx = next_pipe.x - bird_x
        dy = next_pipe.gap_y - bird_y
    return (dx, dy, bird_vy)


# === MELHORIA 7: Features aprimoradas com predição multi-step e histórico ===
def make_enhanced_features(bird_x: float, bird_y: float, bird_vy: float, pipes, 
                          history_buffer: Optional[deque] = None) -> List[float]:
    """Features melhoradas com predição multi-step e histórico"""
    
    candidates = [p for p in pipes if (p.x + PIPE_WIDTH) >= bird_x]
    candidates.sort(key=lambda p: p.x)
    p1 = candidates[0] if len(candidates) > 0 else None
    p2 = candidates[1] if len(candidates) > 1 else None

    def pipe_values(pipe):
        if pipe is None:
            dx = WIDTH
            gap_y = PLAY_HEIGHT / 2
            pipe_x = bird_x + dx
        else:
            dx = pipe.x - bird_x
            gap_y = pipe.gap_y
            pipe_x = pipe.x
        gap_top = gap_y - PIPE_GAP / 2
        gap_bottom = gap_y + PIPE_GAP / 2
        dx_edge = (pipe_x + PIPE_WIDTH) - bird_x
        time_to_front = dx / PIPE_SPEED if PIPE_SPEED > 0 else 0.0
        time_to_back = dx_edge / PIPE_SPEED if PIPE_SPEED > 0 else 0.0
        max_time = WIDTH / PIPE_SPEED if PIPE_SPEED > 0 else 1.0
        y_no_front = predict_y(bird_y, bird_vy, max(0.0, time_to_front))
        y_no_back = predict_y(bird_y, bird_vy, max(0.0, time_to_back))
        y_jump_front = predict_y(bird_y, JUMP_VELOCITY, max(0.0, time_to_front))
        y_jump_back = predict_y(bird_y, JUMP_VELOCITY, max(0.0, time_to_back))
        margin_no = min(y_no_front - gap_top, gap_bottom - y_no_front)
        margin_jump = min(y_jump_front - gap_top, gap_bottom - y_jump_front)
        return (dx, dx_edge, gap_y, gap_top, gap_bottom, time_to_front, time_to_back, max_time,
                y_no_front, y_no_back, y_jump_front, y_jump_back, margin_no, margin_jump)

    (dx1, dx1_edge, gap_y1, gap_top1, gap_bot1, t1f, t1b, max_t,
     y1_no_f, y1_no_b, y1_j_f, y1_j_b, m1_no, m1_j) = pipe_values(p1)
    (dx2, dx2_edge, gap_y2, gap_top2, gap_bot2, t2f, _, _,
     y2_no_f, _, _, _, m2_no, _) = pipe_values(p2)

    t_apex = max(0.0, -JUMP_VELOCITY / GRAVITY)
    y_apex = predict_y(bird_y, JUMP_VELOCITY, t_apex)
    ceil_clear = bird_y - BIRD_RADIUS
    floor_clear = PLAY_HEIGHT - (bird_y + BIRD_RADIUS)
    apex_clear = y_apex - BIRD_RADIUS

    # Features básicas (25)
    features = [
        norm_signed(dx1, WIDTH),
        norm_signed(dx1_edge, WIDTH),
        norm_signed(gap_y1 - bird_y, PLAY_HEIGHT),
        norm_signed(t1f, max_t),
        norm_signed(t1b, max_t),
        norm_signed(y1_no_f, PLAY_HEIGHT),
        norm_signed(y1_no_b, PLAY_HEIGHT),
        norm_signed(y1_j_f, PLAY_HEIGHT),
        norm_signed(y1_j_b, PLAY_HEIGHT),
        norm_signed(m1_no, PLAY_HEIGHT),
        norm_signed(m1_j, PLAY_HEIGHT),
        norm_unit_to_signed(gap_top1, PLAY_HEIGHT),
        norm_unit_to_signed(gap_bot1, PLAY_HEIGHT),
        norm_signed(dx2, WIDTH),
        norm_signed(gap_y2 - bird_y, PLAY_HEIGHT),
        norm_signed(t2f, max_t),
        norm_signed(y2_no_f, PLAY_HEIGHT),
        norm_signed(m2_no, PLAY_HEIGHT),
        norm_signed(bird_vy, TERMINAL_VELOCITY),
        norm_unit_to_signed(bird_y, PLAY_HEIGHT),
        norm_signed(ceil_clear, PLAY_HEIGHT),
        norm_signed(floor_clear, PLAY_HEIGHT),
        norm_signed(apex_clear, PLAY_HEIGHT),
        norm_unit_to_signed(PIPE_GAP, PLAY_HEIGHT),
        norm_unit_to_signed(PIPE_SPEED, 400.0),
    ]
    
    # === NOVO: Predições multi-step (6 features) ===
    prediction_times = [0.15, 0.3, 0.5]
    for t in prediction_times:
        future_y_no = predict_y(bird_y, bird_vy, t)
        future_y_jump = predict_y(bird_y, JUMP_VELOCITY, t)
        features.append(norm_signed(future_y_no, PLAY_HEIGHT))
        features.append(norm_signed(future_y_jump, PLAY_HEIGHT))
    
    # === NOVO: Features de histórico (6 features) ===
    if history_buffer and len(history_buffer) >= 2:
        # Velocidade anterior
        prev_vy = history_buffer[-2].vy if len(history_buffer) >= 2 else bird_vy
        features.append(norm_signed(prev_vy, TERMINAL_VELOCITY))
        
        # Aceleração (mudança de velocidade)
        acceleration = bird_vy - prev_vy
        features.append(norm_signed(acceleration, 1000.0))
        
        # Posição anterior dy
        prev_dy = history_buffer[-2].dy if len(history_buffer) >= 2 else 0.0
        features.append(norm_signed(prev_dy, PLAY_HEIGHT))
        
        # Mudança em dy
        dy_change = (gap_y1 - bird_y) - prev_dy
        features.append(norm_signed(dy_change, PLAY_HEIGHT))
        
        # Tendência de velocidade (média das últimas 3)
        recent_vys = [obs.vy for obs in list(history_buffer)[-3:]]
        avg_vy = sum(recent_vys) / len(recent_vys) if recent_vys else bird_vy
        features.append(norm_signed(avg_vy, TERMINAL_VELOCITY))
        
        # Tendência de dy
        recent_dys = [obs.dy for obs in list(history_buffer)[-3:]]
        avg_dy = sum(recent_dys) / len(recent_dys) if recent_dys else 0.0
        features.append(norm_signed(avg_dy, PLAY_HEIGHT))
    else:
        # Preencher com zeros se não houver histórico
        features.extend([0.0] * 6)
    
    # Total: 25 (básicas) + 6 (multi-step) + 6 (histórico) = 37 features
    return features


# Backward compatibility: usar features antigas se solicitado
def make_features(bird_x: float, bird_y: float, bird_vy: float, pipes) -> List[float]:
    """Features originais (mantida para compatibilidade)"""
    return make_enhanced_features(bird_x, bird_y, bird_vy, pipes, None)[:25]


def heuristic_action(dx: float, dy: float, vy: float, bird_y: float) -> int:
    if dx > 180:
        return 1 if vy > 260 else 0
    if dy < -20:
        return 1 if vy > -220 else 0
    if dy > 35:
        return 0
    return 1 if vy > 320 else 0


def guided_action(obs) -> int:
    if (obs.bird_y - BIRD_RADIUS) < 20 and obs.vy < 0:
        return 0
    t = obs.dx / PIPE_SPEED if PIPE_SPEED > 0 else 0.0
    t = max(0.0, t)
    pred_y = predict_y(obs.bird_y, obs.vy, t)
    gap_y = obs.bird_y + obs.dy
    gap_top = gap_y - PIPE_GAP / 2 + BIRD_RADIUS
    gap_bottom = gap_y + PIPE_GAP / 2 - BIRD_RADIUS
    pred_jump = predict_y(obs.bird_y, JUMP_VELOCITY, t)
    t_apex = max(0.0, -JUMP_VELOCITY / GRAVITY)
    y_apex = predict_y(obs.bird_y, JUMP_VELOCITY, t_apex)
    if (y_apex - BIRD_RADIUS) < 10:
        return 0
    # Se o gap estiver bem abaixo, evite flapar para permitir descer
    if obs.dy > PIPE_GAP * 0.3:
        return 0
    if obs.dx > 240:
        return 1 if obs.vy > 260 else 0
    if pred_y > gap_bottom:
        return 1
    if pred_y < gap_top:
        return 0
    if pred_jump < gap_top:
        return 0
    if pred_jump > gap_bottom:
        return 1
    return 1 if obs.vy > 320 else 0


# === MELHORIA 8: Sistema de recompensas aprimorado ===
def enhanced_reward(bird_y: float, bird_vy: float, dy: float, action: int, 
                   scored: bool, dead: bool, next_pipe) -> float:
    """Sistema de recompensas mais denso e específico"""
    if dead:
        return AI_DEAD_PENALTY
    
    reward = AI_STEP_REWARD  # Recompensa base por sobreviver
    
    # Recompensa GRANDE por passar cano
    if scored:
        reward += AI_PIPE_REWARD
    
    # Recompensa por manter-se perto do centro do gap
    if PIPE_GAP > 0:
        gap_center_distance = abs(dy)
        
        # Recompensa escalonada baseada na distância do centro
        if gap_center_distance < PIPE_GAP * 0.15:  # Muito perto do centro
            reward += AI_CENTER_REWARD * 1.5
        elif gap_center_distance < PIPE_GAP * 0.3:  # Próximo do centro
            reward += AI_CENTER_REWARD
        elif gap_center_distance < PIPE_GAP * 0.45:  # Razoável
            reward += AI_CENTER_REWARD * 0.3
    
    # Penalidade por velocidade excessiva (muito rápido é perigoso)
    if abs(bird_vy) > TERMINAL_VELOCITY * 0.7:
        reward -= 0.05
    
    # === NOVO: Recompensa por preparação antes do cano ===
    if next_pipe:
        dx_to_pipe = next_pipe.x - BIRD_X
        # Zona de preparação: 100-200 pixels antes do cano
        if 100 < dx_to_pipe < 200:
            ideal_y = next_pipe.gap_y
            distance_from_ideal = abs(bird_y - ideal_y)
            
            if distance_from_ideal < 30:  # Muito bem posicionado
                reward += AI_PREP_REWARD
            elif distance_from_ideal < 60:  # Razoavelmente posicionado
                reward += AI_PREP_REWARD * 0.5
    
    # Penalidade suave por pular desnecessariamente
    # (só penalizar se não estava caindo rápido)
    if action == 1 and bird_vy < 200:
        reward += AI_FLAP_PENALTY
    
    # === NOVO: Recompensa por velocidade controlada ===
    if -300 < bird_vy < 300:
        reward += AI_SPEED_REWARD
    
    return reward


# Backward compatibility
def shaped_reward(dy: float, action: int, scored: bool, dead: bool) -> float:
    """Recompensa antiga (mantida para compatibilidade)"""
    if dead:
        return AI_DEAD_PENALTY
    reward = AI_STEP_REWARD
    if scored:
        reward += AI_PIPE_REWARD
    if PIPE_GAP > 0:
        closeness = 1.0 - min(abs(dy) / (PIPE_GAP / 2), 1.0)
        reward += AI_CENTER_REWARD * closeness
    if action == 1:
        reward += AI_FLAP_PENALTY
    return reward


# ------------------------------
# Entidades do jogo
# ------------------------------

@dataclass
class Bird:
    x: float
    y: float
    r: int
    vy: float = 0.0

    def reset(self, y: float) -> None:
        self.y = y
        self.vy = 0.0

    def flap(self) -> None:
        self.vy = JUMP_VELOCITY

    def update(self, dt: float) -> None:
        self.vy += GRAVITY * dt
        self.vy = min(self.vy, TERMINAL_VELOCITY)
        self.y += self.vy * dt


@dataclass
class PipePair:
    top_id: int
    bottom_id: int
    x: float = 0.0
    gap_y: float = 0.0
    scored: bool = False
    active: bool = False

    def reset(self, x: float, gap_y: float) -> None:
        self.x = x
        self.gap_y = gap_y
        self.scored = False
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def update(self, dt: float) -> None:
        self.x -= PIPE_SPEED * dt

    def offscreen(self) -> bool:
        return (self.x + PIPE_WIDTH) < 0


# ------------------------------
# IA: simulador e agente
# ------------------------------

@dataclass
class AIObs:
    state: Tuple[int, int, int]
    features: List[float]
    dx: float
    dy: float
    vy: float
    bird_y: float


def make_obs(bird_x: float, bird_y: float, bird_vy: float, pipes,
             history_buffer: Optional[deque] = None, use_enhanced: bool = True) -> AIObs:
    """Criar observação com opção de usar features aprimoradas"""
    dx, dy, vy = state_values_from_pipes(bird_x, bird_y, bird_vy, pipes)
    
    if use_enhanced:
        features = make_enhanced_features(bird_x, bird_y, bird_vy, pipes, history_buffer)
    else:
        features = make_features(bird_x, bird_y, bird_vy, pipes)
    
    return AIObs(
        state=make_state(dx, dy, vy),
        features=features,
        dx=dx,
        dy=dy,
        vy=vy,
        bird_y=bird_y,
    )


@dataclass
class SimPipe:
    x: float
    gap_y: float
    scored: bool = False


class FlappySim:
    def __init__(self, use_enhanced_features: bool = True) -> None:
        self.bird_y = HEIGHT * 0.45
        self.bird_vy = 0.0
        self.pipes: List[SimPipe] = []
        self.time_since_pipe = 0.0
        self.score = 0
        self.use_enhanced_features = use_enhanced_features
        # === NOVO: Buffer de histórico ===
        self.history_buffer: deque = deque(maxlen=HISTORY_BUFFER_SIZE)

    def reset(self, episode: int = 0) -> AIObs:
        self.bird_y = HEIGHT * 0.45
        self.bird_vy = 0.0
        self.pipes = []
        self.time_since_pipe = 0.0
        self.score = 0
        self.history_buffer.clear()
        
        assist_prob = assist_prob_for_episode(episode)
        gap_y = sample_gap_y(self.bird_y, assist_prob)
        self.spawn_pipe(x=WIDTH * 0.6, gap_y=gap_y)
        
        obs = self.get_obs()
        # Inicializar histórico com a primeira observação
        self.history_buffer.append(obs)
        return obs

    def get_obs(self) -> AIObs:
        return make_obs(BIRD_X, self.bird_y, self.bird_vy, self.pipes, 
                       self.history_buffer, self.use_enhanced_features)

    def spawn_pipe(self, x: Optional[float] = None, gap_y: Optional[int] = None,
                   assist_prob: float = 0.0) -> None:
        if x is None:
            x = WIDTH + PIPE_WIDTH
        if gap_y is None:
            gap_y = sample_gap_y(self.bird_y, assist_prob)
        self.pipes.append(SimPipe(x=x, gap_y=gap_y))

    def update_pipes(self, dt: float) -> None:
        for pipe in self.pipes:
            pipe.x -= PIPE_SPEED * dt
        self.pipes = [p for p in self.pipes if (p.x + PIPE_WIDTH) >= 0]

    def check_collisions(self) -> bool:
        if (self.bird_y + BIRD_RADIUS) >= (HEIGHT - FLOOR_HEIGHT):
            return True
        if (self.bird_y - BIRD_RADIUS) <= 0:
            return True

        for pipe in self.pipes:
            top_height = pipe.gap_y - PIPE_GAP / 2
            bottom_y = pipe.gap_y + PIPE_GAP / 2

            if circle_rect_collision(BIRD_X, self.bird_y, BIRD_RADIUS,
                                     pipe.x, 0, PIPE_WIDTH, top_height):
                return True

            if circle_rect_collision(BIRD_X, self.bird_y, BIRD_RADIUS,
                                     pipe.x, bottom_y, PIPE_WIDTH, HEIGHT - FLOOR_HEIGHT - bottom_y):
                return True

        return False

    def step(self, action: int, dt: float = AI_DECISION_DT, assist_prob: float = 0.0) -> Tuple[AIObs, float, bool]:
        if action == 1:
            self.bird_vy = JUMP_VELOCITY

        self.bird_vy += GRAVITY * dt
        self.bird_vy = min(self.bird_vy, TERMINAL_VELOCITY)
        self.bird_y += self.bird_vy * dt

        self.time_since_pipe += dt
        if self.time_since_pipe >= PIPE_INTERVAL:
            self.time_since_pipe -= PIPE_INTERVAL
            self.spawn_pipe(assist_prob=assist_prob)

        self.update_pipes(dt)

        scored = False
        for pipe in self.pipes:
            if not pipe.scored and (BIRD_X > pipe.x + PIPE_WIDTH):
                pipe.scored = True
                self.score += 1
                scored = True

        done = self.check_collisions()
        
        obs = self.get_obs()
        # Atualizar histórico
        self.history_buffer.append(obs)
        
        # Usar sistema de recompensas aprimorado
        next_pipe = get_next_pipe(self.pipes, BIRD_X)
        reward = enhanced_reward(self.bird_y, self.bird_vy, obs.dy, action, scored, done, next_pipe)

        return obs, reward, done


def get_q(qtable: Dict[Tuple[int, int, int], List[float]], state: Tuple[int, int, int]) -> List[float]:
    if state not in qtable:
        qtable[state] = [0.0, 0.0]
    return qtable[state]


def random_action() -> int:
    return 1 if random.random() < AI_RANDOM_FLAP_PROB else 0


def select_action(qtable: Dict[Tuple[int, int, int], List[float]],
                  obs: AIObs,
                  epsilon: float) -> int:
    if random.random() < epsilon:
        return random_action()
    if obs.state not in qtable:
        return heuristic_action(obs.dx, obs.dy, obs.vy, obs.bird_y)
    q = qtable[obs.state]
    if q[0] == q[1]:
        return heuristic_action(obs.dx, obs.dy, obs.vy, obs.bird_y)
    return 1 if q[1] > q[0] else 0


class QAgent:
    label = "Q"
    decision_dt = AI_DECISION_DT

    def __init__(self, qtable: Dict[Tuple[int, int, int], List[float]], epsilon: float = 0.0) -> None:
        self.qtable = qtable
        self.epsilon = epsilon
        self.alpha = AI_ALPHA
        self.gamma = AI_GAMMA

    def act(self, obs: AIObs) -> int:
        return select_action(self.qtable, obs, self.epsilon)

    def act_from_game(self, bird: Bird, pipes: List[PipePair]) -> int:
        obs = make_obs(bird.x, bird.y, bird.vy, pipes, None, False)
        return self.act(obs)

    def learn(self, obs: AIObs, action: int,
              reward: float, next_obs: AIObs, done: bool) -> None:
        q = get_q(self.qtable, obs.state)
        if done:
            target = reward
        else:
            q_next = get_q(self.qtable, next_obs.state)
            target = reward + self.gamma * max(q_next)
        q[action] += self.alpha * (target - q[action])


# === MELHORIA 9: Prioritized Experience Replay ===
class PrioritizedReplayBuffer:
    """Replay buffer com priorização baseada em erro TD"""
    
    def __init__(self, capacity: int, alpha: float = NN_REPLAY_ALPHA) -> None:
        require_numpy()
        self.capacity = capacity
        self.alpha = alpha  # Quanto priorizar (0 = uniform, 1 = full priority)
        self.buffer: List[Tuple[List[float], int, float, List[float], bool]] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state: List[float], action: int, reward: float,
            next_state: List[float], done: bool, priority: Optional[float] = None) -> None:
        """Adiciona experiência com prioridade"""
        data = (state, action, reward, next_state, done)
        
        # Usar prioridade máxima para novas experiências (garante que sejam amostradas)
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Amostra com base nas prioridades"""
        if len(self.buffer) == 0:
            return None
        
        # Calcular probabilidades baseadas em prioridades
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Amostrar índices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calcular importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalizar
        
        # Coletar amostras
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities) -> None:
        """Atualiza prioridades após treinamento"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


# Classe antiga mantida para compatibilidade
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[List[float], int, float, List[float], bool]] = []
        self.pos = 0

    def add(self, state: List[float], action: int, reward: float,
            next_state: List[float], done: bool) -> None:
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idxs))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNNet:
    def __init__(self, sizes: Tuple[int, ...]) -> None:
        require_numpy()
        self.sizes = sizes
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            w = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / in_size)
            b = np.zeros(out_size, dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

        self._cache_a = []
        self._cache_z = []

    @staticmethod
    def _act(x):
        return np.where(x > 0, x, 0.1 * x)

    @staticmethod
    def _act_grad(x):
        return np.where(x > 0, 1.0, 0.1)

    def forward(self, x, cache: bool = False):
        a = x
        if cache:
            self._cache_a = [a]
            self._cache_z = []
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self._act(z)
            if cache:
                self._cache_z.append(z)
                self._cache_a.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        if cache:
            self._cache_z.append(z)
        return z

    def predict(self, features: List[float]) -> List[float]:
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        q = self.forward(x, cache=False)
        return q[0].tolist()

    def train_on_batch(self, x, target, lr: float, weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Retorna loss e erros TD (para prioritized replay)"""
        pred = self.forward(x, cache=True)
        diff = pred - target
        
        # Aplicar importance sampling weights se fornecidos
        if weights is not None:
            diff = diff * weights.reshape(-1, 1)
        
        loss = float((diff * diff).mean())
        
        # Calcular erros TD para atualização de prioridades
        td_errors = np.abs(diff).max(axis=1)
        
        d_out = (2.0 * diff) / diff.size

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        d = d_out
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self._cache_a[i]
            grads_w[i] = a_prev.T @ d
            grads_b[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.weights[i].T) * self._act_grad(self._cache_z[i - 1])

        self._apply_adam(grads_w, grads_b, lr)
        return loss, td_errors

    def _apply_adam(self, grads_w, grads_b, lr: float) -> None:
        self.t += 1
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8

        total_norm = 0.0
        for gw in grads_w:
            total_norm += float((gw * gw).sum())
        for gb in grads_b:
            total_norm += float((gb * gb).sum())
        total_norm = total_norm ** 0.5
        if total_norm > NN_MAX_GRAD_NORM:
            scale = NN_MAX_GRAD_NORM / max(total_norm, 1e-6)
            grads_w = [gw * scale for gw in grads_w]
            grads_b = [gb * scale for gb in grads_b]

        for i in range(len(self.weights)):
            self.m_w[i] = b1 * self.m_w[i] + (1 - b1) * grads_w[i]
            self.v_w[i] = b2 * self.v_w[i] + (1 - b2) * (grads_w[i] ** 2)
            self.m_b[i] = b1 * self.m_b[i] + (1 - b1) * grads_b[i]
            self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * (grads_b[i] ** 2)

            m_w_hat = self.m_w[i] / (1 - b1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - b2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - b1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - b2 ** self.t)

            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def get_weights(self):
        return [w.copy() for w in self.weights], [b.copy() for b in self.biases]

    def set_weights(self, weights, biases) -> None:
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]


# === MELHORIA 10: Dueling DQN Architecture ===
class DuelingDQNNet:
    """Arquitetura Dueling DQN: separa V(s) e A(s,a)"""
    
    def __init__(self, sizes: Tuple[int, ...]) -> None:
        require_numpy()
        self.sizes = sizes
        
        # Camadas compartilhadas (até penúltima)
        self.shared_weights = []
        self.shared_biases = []
        
        # Dividir em shared + value/advantage streams
        shared_layers = sizes[:-1]  # Todas menos a última
        
        for i in range(len(shared_layers) - 1):
            in_size = shared_layers[i]
            out_size = shared_layers[i + 1]
            w = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2.0 / in_size)
            b = np.zeros(out_size, dtype=np.float32)
            self.shared_weights.append(w)
            self.shared_biases.append(b)
        
        # Value stream: prediz V(s) (1 valor)
        last_hidden = shared_layers[-1]
        self.value_w = np.random.randn(last_hidden, 1).astype(np.float32) * np.sqrt(2.0 / last_hidden)
        self.value_b = np.zeros(1, dtype=np.float32)
        
        # Advantage stream: prediz A(s,a) (2 valores para 2 ações)
        n_actions = 2
        self.advantage_w = np.random.randn(last_hidden, n_actions).astype(np.float32) * np.sqrt(2.0 / last_hidden)
        self.advantage_b = np.zeros(n_actions, dtype=np.float32)
        
        # Adam optimizer params
        self.m_sw = [np.zeros_like(w) for w in self.shared_weights]
        self.v_sw = [np.zeros_like(w) for w in self.shared_weights]
        self.m_sb = [np.zeros_like(b) for b in self.shared_biases]
        self.v_sb = [np.zeros_like(b) for b in self.shared_biases]
        
        self.m_vw = np.zeros_like(self.value_w)
        self.v_vw = np.zeros_like(self.value_w)
        self.m_vb = np.zeros_like(self.value_b)
        self.v_vb = np.zeros_like(self.value_b)
        
        self.m_aw = np.zeros_like(self.advantage_w)
        self.v_aw = np.zeros_like(self.advantage_w)
        self.m_ab = np.zeros_like(self.advantage_b)
        self.v_ab = np.zeros_like(self.advantage_b)
        
        self.t = 0
        self._cache_shared = []
        self._cache_z_shared = []

    @staticmethod
    def _act(x):
        return np.where(x > 0, x, 0.1 * x)

    @staticmethod
    def _act_grad(x):
        return np.where(x > 0, 1.0, 0.1)

    def forward(self, x, cache: bool = False):
        # Forward através das camadas compartilhadas
        a = x
        if cache:
            self._cache_shared = [a]
            self._cache_z_shared = []
        
        for i in range(len(self.shared_weights)):
            z = a @ self.shared_weights[i] + self.shared_biases[i]
            a = self._act(z)
            if cache:
                self._cache_z_shared.append(z)
                self._cache_shared.append(a)
        
        # Value stream
        value = a @ self.value_w + self.value_b  # (batch, 1)
        
        # Advantage stream
        advantages = a @ self.advantage_w + self.advantage_b  # (batch, n_actions)
        
        # Combinar: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        # Isso garante identificabilidade: V(s) é único
        q_values = value + (advantages - advantages.mean(axis=1, keepdims=True))
        
        return q_values

    def predict(self, features: List[float]) -> List[float]:
        x = np.array(features, dtype=np.float32).reshape(1, -1)
        q = self.forward(x, cache=False)
        return q[0].tolist()

    def train_on_batch(self, x, target, lr: float, weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        pred = self.forward(x, cache=True)
        diff = pred - target
        
        if weights is not None:
            diff = diff * weights.reshape(-1, 1)
        
        loss = float((diff * diff).mean())
        td_errors = np.abs(diff).max(axis=1)
        
        # Backprop
        d_out = (2.0 * diff) / diff.size
        
        # Gradientes para advantage stream
        grad_adv_w = self._cache_shared[-1].T @ d_out
        grad_adv_b = d_out.sum(axis=0)
        
        # Gradientes para value stream
        # d_out precisa ser propagado considerando a subtração da média
        d_value = d_out.sum(axis=1, keepdims=True)  # Soma sobre ações
        grad_val_w = self._cache_shared[-1].T @ d_value
        grad_val_b = d_value.sum(axis=0)
        
        # Gradientes para camadas compartilhadas
        # Propagar de ambas streams
        d_shared_from_adv = d_out @ self.advantage_w.T
        d_shared_from_val = d_value @ self.value_w.T
        d_shared = (d_shared_from_adv + d_shared_from_val) * self._act_grad(self._cache_z_shared[-1])
        
        grads_sw = [None] * len(self.shared_weights)
        grads_sb = [None] * len(self.shared_biases)
        
        for i in range(len(self.shared_weights) - 1, -1, -1):
            a_prev = self._cache_shared[i]
            grads_sw[i] = a_prev.T @ d_shared
            grads_sb[i] = d_shared.sum(axis=0)
            if i > 0:
                d_shared = (d_shared @ self.shared_weights[i].T) * self._act_grad(self._cache_z_shared[i - 1])
        
        # Aplicar Adam
        self._apply_adam_dueling(grads_sw, grads_sb, grad_val_w, grad_val_b, 
                                grad_adv_w, grad_adv_b, lr)
        
        return loss, td_errors

    def _apply_adam_dueling(self, grads_sw, grads_sb, grad_vw, grad_vb, 
                           grad_aw, grad_ab, lr: float) -> None:
        self.t += 1
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        
        # Gradient clipping
        all_grads = grads_sw + grads_sb + [grad_vw, grad_vb, grad_aw, grad_ab]
        total_norm = sum(float((g * g).sum()) for g in all_grads) ** 0.5
        
        if total_norm > NN_MAX_GRAD_NORM:
            scale = NN_MAX_GRAD_NORM / max(total_norm, 1e-6)
            grads_sw = [g * scale for g in grads_sw]
            grads_sb = [g * scale for g in grads_sb]
            grad_vw *= scale
            grad_vb *= scale
            grad_aw *= scale
            grad_ab *= scale
        
        # Atualizar shared weights
        for i in range(len(self.shared_weights)):
            self.m_sw[i] = b1 * self.m_sw[i] + (1 - b1) * grads_sw[i]
            self.v_sw[i] = b2 * self.v_sw[i] + (1 - b2) * (grads_sw[i] ** 2)
            self.m_sb[i] = b1 * self.m_sb[i] + (1 - b1) * grads_sb[i]
            self.v_sb[i] = b2 * self.v_sb[i] + (1 - b2) * (grads_sb[i] ** 2)
            
            m_w_hat = self.m_sw[i] / (1 - b1 ** self.t)
            v_w_hat = self.v_sw[i] / (1 - b2 ** self.t)
            m_b_hat = self.m_sb[i] / (1 - b1 ** self.t)
            v_b_hat = self.v_sb[i] / (1 - b2 ** self.t)
            
            self.shared_weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.shared_biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        
        # Atualizar value stream
        self.m_vw = b1 * self.m_vw + (1 - b1) * grad_vw
        self.v_vw = b2 * self.v_vw + (1 - b2) * (grad_vw ** 2)
        self.m_vb = b1 * self.m_vb + (1 - b1) * grad_vb
        self.v_vb = b2 * self.v_vb + (1 - b2) * (grad_vb ** 2)
        
        m_vw_hat = self.m_vw / (1 - b1 ** self.t)
        v_vw_hat = self.v_vw / (1 - b2 ** self.t)
        m_vb_hat = self.m_vb / (1 - b1 ** self.t)
        v_vb_hat = self.v_vb / (1 - b2 ** self.t)
        
        self.value_w -= lr * m_vw_hat / (np.sqrt(v_vw_hat) + eps)
        self.value_b -= lr * m_vb_hat / (np.sqrt(v_vb_hat) + eps)
        
        # Atualizar advantage stream
        self.m_aw = b1 * self.m_aw + (1 - b1) * grad_aw
        self.v_aw = b2 * self.v_aw + (1 - b2) * (grad_aw ** 2)
        self.m_ab = b1 * self.m_ab + (1 - b1) * grad_ab
        self.v_ab = b2 * self.v_ab + (1 - b2) * (grad_ab ** 2)
        
        m_aw_hat = self.m_aw / (1 - b1 ** self.t)
        v_aw_hat = self.v_aw / (1 - b2 ** self.t)
        m_ab_hat = self.m_ab / (1 - b1 ** self.t)
        v_ab_hat = self.v_ab / (1 - b2 ** self.t)
        
        self.advantage_w -= lr * m_aw_hat / (np.sqrt(v_aw_hat) + eps)
        self.advantage_b -= lr * m_ab_hat / (np.sqrt(v_ab_hat) + eps)

    def get_weights(self):
        """Retorna todos os pesos em formato compatível"""
        return {
            'shared': ([w.copy() for w in self.shared_weights], 
                      [b.copy() for b in self.shared_biases]),
            'value': (self.value_w.copy(), self.value_b.copy()),
            'advantage': (self.advantage_w.copy(), self.advantage_b.copy())
        }

    def set_weights(self, weights_dict) -> None:
        """Define todos os pesos a partir de dicionário"""
        shared_w, shared_b = weights_dict['shared']
        self.shared_weights = [w.copy() for w in shared_w]
        self.shared_biases = [b.copy() for b in shared_b]
        
        self.value_w = weights_dict['value'][0].copy()
        self.value_b = weights_dict['value'][1].copy()
        
        self.advantage_w = weights_dict['advantage'][0].copy()
        self.advantage_b = weights_dict['advantage'][1].copy()


# === MELHORIA 11: Double DQN Agent ===
class DoubleDQNAgent:
    """Agente DQN aprimorado com Double DQN e Prioritized Replay"""
    label = "NN-Enhanced"
    decision_dt = NN_DECISION_DT

    def __init__(self, layers: Tuple[int, ...] = NN_LAYERS, use_dueling: bool = False) -> None:
        require_numpy()
        self.layers = layers
        self.device_name = "numpy"
        self.use_dueling = use_dueling
        
        # Criar redes
        if use_dueling:
            self.net = DuelingDQNNet(layers)
            self.best_net = DuelingDQNNet(layers)
            self.target_net = DuelingDQNNet(layers)
        else:
            self.net = DQNNet(layers)
            self.best_net = DQNNet(layers)
            self.target_net = DQNNet(layers)
        
        self.target_net.set_weights(self.net.get_weights())
        self.best_net.set_weights(self.net.get_weights())
        
        # Usar Prioritized Replay
        self.replay = PrioritizedReplayBuffer(NN_REPLAY_SIZE)
        self.beta = NN_REPLAY_BETA_START
        self.beta_increment = (1.0 - NN_REPLAY_BETA_START) / NN_REPLAY_BETA_FRAMES
        
        self.epsilon = NN_EPS_START
        self.gamma = NN_GAMMA
        self.lr = NN_LR
        self.learn_steps = 0
        self.best_score = 0
        self.use_best = False
        self.guide_only = False
        self.train_steps = NN_TRAIN_STEPS

    def act(self, obs: AIObs) -> int:
        if self.guide_only:
            return guided_action(obs)
        eps = 0.0 if self.use_best else self.epsilon
        if random.random() < eps:
            if random.random() < NN_GUIDE_PROB:
                return guided_action(obs)
            return random_action()
        net = self.best_net if self.use_best else self.net
        q = net.predict(obs.features)
        action = 1 if q[1] > q[0] else 0
        
        # Safety check
        t_apex = max(0.0, -JUMP_VELOCITY / GRAVITY)
        y_apex = predict_y(obs.bird_y, JUMP_VELOCITY, t_apex)
        if action == 1 and ((obs.bird_y - BIRD_RADIUS) < 16 or (y_apex - BIRD_RADIUS) < 8):
            action = 0
        if action == 1 and obs.dy > PIPE_GAP * 0.3:
            action = 0
        return action

    def learn(self, obs: AIObs, action: int, reward: float, next_obs: AIObs, done: bool) -> Optional[float]:
        # Adicionar ao replay buffer (prioridade será calculada depois)
        self.replay.add(obs.features, action, reward, next_obs.features, done)
        
        if len(self.replay) < max(NN_TRAIN_START, NN_BATCH_SIZE):
            return None
        
        # Incrementar beta (importance sampling weight)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        loss = None
        for _ in range(self.train_steps):
            # Sample com prioridades
            sample_result = self.replay.sample(NN_BATCH_SIZE, self.beta)
            if sample_result is None:
                continue
            
            states, actions, rewards, next_states, dones, indices, is_weights = sample_result
            
            x = np.array(states, dtype=np.float32)
            x_next = np.array(next_states, dtype=np.float32)
            is_weights_np = np.array(is_weights, dtype=np.float32)
            
            # === Double DQN: usar net principal para selecionar, target para avaliar ===
            q_pred = self.net.forward(x, cache=False)
            q_next_main = self.net.forward(x_next, cache=False)  # Rede principal seleciona
            q_next_target = self.target_net.forward(x_next, cache=False)  # Target avalia
            
            target = q_pred.copy()
            
            for i in range(NN_BATCH_SIZE):
                if dones[i]:
                    target[i, actions[i]] = rewards[i]
                else:
                    # Double DQN: selecionar com main, avaliar com target
                    best_action = int(q_next_main[i].argmax())
                    target[i, actions[i]] = rewards[i] + self.gamma * q_next_target[i, best_action]
            
            # Treinar com importance sampling weights
            loss, td_errors = self.net.train_on_batch(x, target, self.lr, is_weights_np)
            
            # Atualizar prioridades no replay buffer
            new_priorities = td_errors + 1e-5  # Pequeno epsilon para evitar zero
            self.replay.update_priorities(indices, new_priorities)
            
            self.learn_steps += 1
            if (self.learn_steps % NN_TARGET_UPDATE) == 0:
                self.target_net.set_weights(self.net.get_weights())
        
        return loss

    def decay_epsilon(self, success: bool = False, failed: bool = False) -> None:
        if failed:
            # Falhou: reduzir exploracao gradualmente (sem piso fixo que trava)
            self.epsilon = max(NN_EPS_MIN, self.epsilon * NN_EPS_DECAY_FAIL)
            return
        decay = NN_EPS_DECAY * (NN_EPS_SUCCESS_DECAY if success else 1.0)
        min_eps = NN_EPS_MIN_SUCCESS if success else NN_EPS_MIN
        self.epsilon = max(min_eps, self.epsilon * decay)

    def update_best(self, score: int) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_net.set_weights(self.net.get_weights())


# Manter agentes antigos para compatibilidade
class DQNAgent:
    label = "NN"
    decision_dt = NN_DECISION_DT

    def __init__(self, layers: Tuple[int, ...] = NN_LAYERS) -> None:
        require_numpy()
        self.layers = layers
        self.device_name = "numpy"
        self.net = DQNNet(layers)
        self.best_net = DQNNet(layers)
        self.target_net = DQNNet(layers)
        self.target_net.set_weights(*self.net.get_weights())
        self.best_net.set_weights(*self.net.get_weights())
        self.replay = ReplayBuffer(NN_REPLAY_SIZE)
        self.epsilon = NN_EPS_START
        self.gamma = NN_GAMMA
        self.lr = NN_LR
        self.learn_steps = 0
        self.best_score = 0
        self.use_best = False
        self.guide_only = False
        self.train_steps = NN_TRAIN_STEPS

    def act(self, obs: AIObs) -> int:
        if self.guide_only:
            return guided_action(obs)
        eps = 0.0 if self.use_best else self.epsilon
        if random.random() < eps:
            if random.random() < NN_GUIDE_PROB:
                return guided_action(obs)
            return random_action()
        net = self.best_net if self.use_best else self.net
        q = net.predict(obs.features)
        action = 1 if q[1] > q[0] else 0
        t_apex = max(0.0, -JUMP_VELOCITY / GRAVITY)
        y_apex = predict_y(obs.bird_y, JUMP_VELOCITY, t_apex)
        if action == 1 and ((obs.bird_y - BIRD_RADIUS) < 16 or (y_apex - BIRD_RADIUS) < 8):
            action = 0
        if action == 1 and obs.dy > PIPE_GAP * 0.3:
            action = 0
        return action

    def learn(self, obs: AIObs, action: int, reward: float, next_obs: AIObs, done: bool) -> Optional[float]:
        self.replay.add(obs.features, action, reward, next_obs.features, done)
        if len(self.replay) < max(NN_TRAIN_START, NN_BATCH_SIZE):
            return None
        loss = None
        for _ in range(self.train_steps):
            states, actions, rewards, next_states, dones = self.replay.sample(NN_BATCH_SIZE)
            x = np.array(states, dtype=np.float32)
            x_next = np.array(next_states, dtype=np.float32)
            q_pred = self.net.forward(x, cache=False)
            q_next = self.target_net.forward(x_next, cache=False)
            target = q_pred.copy()

            for i in range(NN_BATCH_SIZE):
                if dones[i]:
                    target[i, actions[i]] = rewards[i]
                else:
                    target[i, actions[i]] = rewards[i] + self.gamma * float(q_next[i].max())

            loss, _ = self.net.train_on_batch(x, target, self.lr)
            self.learn_steps += 1
            if (self.learn_steps % NN_TARGET_UPDATE) == 0:
                self.target_net.set_weights(*self.net.get_weights())
        return loss

    def decay_epsilon(self, success: bool = False, failed: bool = False) -> None:
        if failed:
            # Falhou: reduzir exploracao gradualmente (sem piso fixo que trava)
            self.epsilon = max(NN_EPS_MIN, self.epsilon * NN_EPS_DECAY_FAIL)
            return
        decay = NN_EPS_DECAY * (NN_EPS_SUCCESS_DECAY if success else 1.0)
        min_eps = NN_EPS_MIN_SUCCESS if success else NN_EPS_MIN
        self.epsilon = max(min_eps, self.epsilon * decay)

    def update_best(self, score: int) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_net.set_weights(*self.net.get_weights())


# [CONTINUA NA PARTE 2...]
# [CONTINUAÇÃO DA PARTE 1...]

# Versões Torch dos agentes melhorados
class TorchDQNNet:
    def __init__(self, sizes: Tuple[int, ...], device) -> None:
        require_torch()
        self.sizes = sizes
        self.device = device
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            w = torch.randn(in_size, out_size, device=device, dtype=torch.float32) * (2.0 / in_size) ** 0.5
            b = torch.zeros(out_size, device=device, dtype=torch.float32)
            self.weights.append(w)
            self.biases.append(b)

        self.m_w = [torch.zeros_like(w) for w in self.weights]
        self.v_w = [torch.zeros_like(w) for w in self.weights]
        self.m_b = [torch.zeros_like(b) for b in self.biases]
        self.v_b = [torch.zeros_like(b) for b in self.biases]
        self.t = 0
        self._cache_a = []
        self._cache_z = []

    @staticmethod
    def _act(x):
        return torch.where(x > 0, x, 0.1 * x)

    @staticmethod
    def _act_grad(x):
        return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, 0.1))

    def forward(self, x, cache: bool = False):
        a = x
        if cache:
            self._cache_a = [a]
            self._cache_z = []
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self._act(z)
            if cache:
                self._cache_z.append(z)
                self._cache_a.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        if cache:
            self._cache_z.append(z)
        return z

    def predict(self, features: List[float]) -> List[float]:
        x = torch.tensor(features, device=self.device, dtype=torch.float32).view(1, -1)
        q = self.forward(x, cache=False)
        return q[0].detach().cpu().tolist()

    def train_on_batch(self, x, target, lr: float, weights=None):
        pred = self.forward(x, cache=True)
        diff = pred - target
        
        if weights is not None:
            diff = diff * weights.view(-1, 1)
        
        loss = float((diff * diff).mean().item())
        td_errors = torch.abs(diff).max(dim=1).values.detach().cpu().numpy()
        
        d_out = (2.0 * diff) / diff.numel()

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        d = d_out
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self._cache_a[i]
            grads_w[i] = a_prev.t() @ d
            grads_b[i] = d.sum(dim=0)
            if i > 0:
                d = (d @ self.weights[i].t()) * self._act_grad(self._cache_z[i - 1])

        self._apply_adam(grads_w, grads_b, lr)
        return loss, td_errors

    def _apply_adam(self, grads_w, grads_b, lr: float) -> None:
        self.t += 1
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8

        total_norm = 0.0
        for gw in grads_w:
            total_norm += float((gw * gw).sum().item())
        for gb in grads_b:
            total_norm += float((gb * gb).sum().item())
        total_norm = total_norm ** 0.5
        if total_norm > NN_MAX_GRAD_NORM:
            scale = NN_MAX_GRAD_NORM / max(total_norm, 1e-6)
            grads_w = [gw * scale for gw in grads_w]
            grads_b = [gb * scale for gb in grads_b]

        for i in range(len(self.weights)):
            self.m_w[i] = b1 * self.m_w[i] + (1 - b1) * grads_w[i]
            self.v_w[i] = b2 * self.v_w[i] + (1 - b2) * (grads_w[i] ** 2)
            self.m_b[i] = b1 * self.m_b[i] + (1 - b1) * grads_b[i]
            self.v_b[i] = b2 * self.v_b[i] + (1 - b2) * (grads_b[i] ** 2)

            m_w_hat = self.m_w[i] / (1 - b1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - b2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - b1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - b2 ** self.t)

            self.weights[i] -= lr * m_w_hat / (torch.sqrt(v_w_hat) + eps)
            self.biases[i] -= lr * m_b_hat / (torch.sqrt(v_b_hat) + eps)

    def get_weights(self):
        return ([w.detach().cpu().numpy() for w in self.weights],
                [b.detach().cpu().numpy() for b in self.biases])

    def set_weights(self, weights, biases) -> None:
        self.weights = [torch.tensor(w, device=self.device, dtype=torch.float32) for w in weights]
        self.biases = [torch.tensor(b, device=self.device, dtype=torch.float32) for b in biases]
        self.m_w = [torch.zeros_like(w) for w in self.weights]
        self.v_w = [torch.zeros_like(w) for w in self.weights]
        self.m_b = [torch.zeros_like(b) for b in self.biases]
        self.v_b = [torch.zeros_like(b) for b in self.biases]


class TorchDQNAgent:
    label = "NN-CUDA"
    decision_dt = NN_DECISION_DT

    def __init__(self, layers: Tuple[int, ...], device) -> None:
        require_torch()
        self.layers = layers
        self.device = device
        self.device_name = str(device)
        self.net = TorchDQNNet(layers, device)
        self.best_net = TorchDQNNet(layers, device)
        self.target_net = TorchDQNNet(layers, device)
        self.target_net.set_weights(*self.net.get_weights())
        self.best_net.set_weights(*self.net.get_weights())
        self.replay = ReplayBuffer(NN_REPLAY_SIZE)
        self.epsilon = NN_EPS_START
        self.gamma = NN_GAMMA
        self.lr = NN_LR
        self.learn_steps = 0
        self.best_score = 0
        self.use_best = False
        self.guide_only = False
        self.train_steps = NN_TRAIN_STEPS

    def act(self, obs: AIObs) -> int:
        if self.guide_only:
            return guided_action(obs)
        eps = 0.0 if self.use_best else self.epsilon
        if random.random() < eps:
            if random.random() < NN_GUIDE_PROB:
                return guided_action(obs)
            return random_action()
        net = self.best_net if self.use_best else self.net
        q = net.predict(obs.features)
        action = 1 if q[1] > q[0] else 0
        t_apex = max(0.0, -JUMP_VELOCITY / GRAVITY)
        y_apex = predict_y(obs.bird_y, JUMP_VELOCITY, t_apex)
        if action == 1 and ((obs.bird_y - BIRD_RADIUS) < 16 or (y_apex - BIRD_RADIUS) < 8):
            action = 0
        if action == 1 and obs.dy > PIPE_GAP * 0.3:
            action = 0
        return action

    def learn(self, obs: AIObs, action: int, reward: float, next_obs: AIObs, done: bool) -> Optional[float]:
        self.replay.add(obs.features, action, reward, next_obs.features, done)
        if len(self.replay) < max(NN_TRAIN_START, NN_BATCH_SIZE):
            return None
        loss = None
        for _ in range(self.train_steps):
            states, actions, rewards, next_states, dones = self.replay.sample(NN_BATCH_SIZE)
            x = torch.tensor(states, device=self.device, dtype=torch.float32)
            x_next = torch.tensor(next_states, device=self.device, dtype=torch.float32)
            q_pred = self.net.forward(x, cache=False)
            q_next = self.target_net.forward(x_next, cache=False)
            target = q_pred.clone()

            actions_t = torch.tensor(actions, device=self.device, dtype=torch.int64)
            rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32)
            max_next = torch.max(q_next, dim=1).values
            target[torch.arange(NN_BATCH_SIZE, device=self.device), actions_t] = rewards_t + (
                1.0 - dones_t
            ) * self.gamma * max_next

            loss, _ = self.net.train_on_batch(x, target, self.lr)
            self.learn_steps += 1
            if (self.learn_steps % NN_TARGET_UPDATE) == 0:
                self.target_net.set_weights(*self.net.get_weights())
        return loss

    def decay_epsilon(self, success: bool = False, failed: bool = False) -> None:
        if failed:
            # Falhou: reduzir exploracao gradualmente (sem piso fixo que trava)
            self.epsilon = max(NN_EPS_MIN, self.epsilon * NN_EPS_DECAY_FAIL)
            return
        decay = NN_EPS_DECAY * (NN_EPS_SUCCESS_DECAY if success else 1.0)
        min_eps = NN_EPS_MIN_SUCCESS if success else NN_EPS_MIN
        self.epsilon = max(min_eps, self.epsilon * decay)

    def update_best(self, score: int) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_net.set_weights(*self.net.get_weights())


# === Funções de salvar/carregar ===
def save_dqn(path: str, agent) -> None:
    data = {
        "layers": agent.layers,
        "weights": agent.net.get_weights(),
        "backend": getattr(agent, "label", "NN"),
        "best_weights": agent.best_net.get_weights() if hasattr(agent, "best_net") else None,
        "use_dueling": getattr(agent, "use_dueling", False),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_dqn(path: str, backend: str = "auto", use_enhanced: bool = True):
    with open(path, "rb") as f:
        data = pickle.load(f)
    layers = data["layers"]
    saved_backend = data.get("backend", "NN")
    use_dueling = data.get("use_dueling", False)
    use_torch = backend == "torch" or (backend == "auto" and saved_backend == "NN-CUDA")
    
    if use_enhanced and not use_torch:
        # Usar versão aprimorada
        agent = DoubleDQNAgent(tuple(layers), use_dueling=use_dueling)
    elif use_torch and torch is not None:
        device = get_torch_device(prefer_cuda=True)
        agent = TorchDQNAgent(tuple(layers), device)
    else:
        require_numpy()
        agent = DQNAgent(tuple(layers))
    
    weights, biases = data["weights"]
    
    # Adaptar carregamento para Dueling DQN
    if use_dueling and isinstance(weights, dict):
        agent.net.set_weights(weights)
        agent.target_net.set_weights(weights)
    else:
        # Compatibilidade com formato antigo
        if hasattr(agent.net, 'set_weights'):
            if isinstance(agent.net, DuelingDQNNet):
                # Converter formato antigo para dueling (se possível)
                pass  # Pode adicionar lógica de conversão aqui
            else:
                agent.net.set_weights(weights, biases)
                agent.target_net.set_weights(weights, biases)
    
    best = data.get("best_weights")
    if best is not None and hasattr(agent, "best_net"):
        if isinstance(best, dict) and use_dueling:
            agent.best_net.set_weights(best)
        elif not isinstance(best, dict):
            best_w, best_b = best
            agent.best_net.set_weights(best_w, best_b)
    
    agent.epsilon = 0.0
    return agent


def train_dqn(episodes: int, max_steps: int, backend: str = "auto", use_enhanced: bool = True):
    """Treinar DQN com opção de usar versão aprimorada"""
    env = FlappySim(use_enhanced_features=use_enhanced)
    
    use_torch = backend == "torch" or (backend == "auto" and torch is not None and torch.cuda.is_available())
    
    if use_enhanced and not use_torch:
        # Usar versão aprimorada com Double DQN e Prioritized Replay
        print("Usando DoubleDQNAgent aprimorado com Prioritized Replay e Dueling DQN")
        agent = DoubleDQNAgent(NN_LAYERS, use_dueling=True)
        device_name = "numpy-enhanced"
    elif use_torch:
        device = get_torch_device(prefer_cuda=True)
        agent = TorchDQNAgent(NN_LAYERS, device)
        device_name = str(device)
    else:
        require_numpy()
        agent = DQNAgent(NN_LAYERS)
        device_name = "numpy"
    
    best_score = 0
    score_history = []

    for ep in range(1, episodes + 1):
        obs = env.reset(episode=ep)
        if len(obs.features) != NN_LAYERS[0]:
            raise SystemExit(f"NN_LAYERS[0] ({NN_LAYERS[0]}) precisa bater com features ({len(obs.features)}). Apague o modelo antigo.")
        ep_reward = 0.0
        loss = None
        assist_prob = assist_prob_for_episode(ep)
        
        # === Curriculum adaptativo ===
        if ep > NN_WARMUP_EPISODES and len(score_history) >= 20:
            avg_score = sum(score_history[-20:]) / 20
            global PIPE_GAP, PIPE_SPEED
            PIPE_GAP, PIPE_SPEED = adaptive_difficulty(avg_score)

        for _ in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, done = env.step(action, dt=NN_DECISION_DT, assist_prob=assist_prob)
            loss = agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            if done:
                break

        score_history.append(env.score)
        if env.score > best_score:
            best_score = env.score

        agent.decay_epsilon(success=env.score >= STABLE_SCORE_ON, failed=env.score <= 0)
        agent.update_best(env.score)
        
        if ep == 1 or ep % 50 == 0:
            loss_str = f"{loss:.4f}" if loss is not None else "n/a"
            avg_score = sum(score_history[-50:]) / min(50, len(score_history))
            print(f"ep {ep:4d} score {env.score:3d} best {best_score:3d} avg {avg_score:5.1f} "
                  f"eps {agent.epsilon:.3f} loss {loss_str} gap {PIPE_GAP:.0f} speed {PIPE_SPEED:.0f} dev {device_name}")

    return agent


def train_qtable(episodes: int, max_steps: int) -> Dict[Tuple[int, int, int], List[float]]:
    env = FlappySim(use_enhanced_features=False)
    qtable: Dict[Tuple[int, int, int], List[float]] = {}
    best_score = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        epsilon = max(AI_EPS_MIN, AI_EPS_START * (AI_EPS_DECAY ** (ep - 1)))

        for _ in range(max_steps):
            action = select_action(qtable, obs, epsilon)

            next_obs, reward, done = env.step(action)
            q = get_q(qtable, obs.state)
            q_next = get_q(qtable, next_obs.state)
            q[action] += AI_ALPHA * (reward + AI_GAMMA * max(q_next) - q[action])

            obs = next_obs
            if done:
                break

        if env.score > best_score:
            best_score = env.score

        if ep == 1 or ep % 50 == 0:
            print(f"ep {ep} score {env.score} best {best_score} eps {epsilon:.3f} q {len(qtable)}")

    return qtable


def save_qtable(path: str, qtable: Dict[Tuple[int, int, int], List[float]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(qtable, f)


def load_qtable(path: str) -> Dict[Tuple[int, int, int], List[float]]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------
# Jogo
# ------------------------------

class FlappyGame:
    def __init__(self, ai_agent: Optional[QAgent] = None) -> None:
        self.root = tk.Tk()
        self.root.title("Flappy Bird - IA Aprimorada")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, highlightthickness=0, bg=BG_COLOR)
        self.canvas.pack()

        self.ai_agent = ai_agent
        self.ai_enabled = ai_agent is not None
        self.ai_train = False
        self.ai_episode = 0
        self.ai_epsilon = AI_EPS_START
        self.ai_save_every = 5
        self.ai_decision_timer = 0.0
        self.ai_decision_dt = getattr(ai_agent, "decision_dt", AI_DECISION_DT)
        self.ai_last_obs: Optional[AIObs] = None
        self.ai_last_action = 0
        self.ai_reward_accum = 0.0
        self.ai_last_loss: Optional[float] = None
        self.speed_mul = TRAIN_SPEED_DEFAULT
        self.score_history: List[int] = []
        self.best_score = 0
        self.auto_turbo = True
        self.auto_stable = True
        self.fail_streak = 0
        self.force_guide_episodes = 0
        
        # === NOVO: Buffer de histórico para o jogo ===
        self.history_buffer: deque = deque(maxlen=HISTORY_BUFFER_SIZE)

        # chao
        self.canvas.create_rectangle(0, HEIGHT - FLOOR_HEIGHT, WIDTH, HEIGHT, fill=FLOOR_COLOR, outline="")

        # passaro
        self.bird = Bird(x=BIRD_X, y=HEIGHT * 0.45, r=BIRD_RADIUS)
        self.bird_id = self.canvas.create_oval(0, 0, 0, 0, fill="#ffcd4d", outline="#e19b2d")
        self._update_bird_draw()
        self.ai_line_id = self.canvas.create_line(-100, -100, -50, -50, fill="#e64a4a", width=2)

        # texto de UI
        self.score = 0
        self.ai_status_id = self.canvas.create_text(WIDTH - 60, 22, text="", fill=TEXT_COLOR,
                                                    font=("Helvetica", 10, "bold"))
        self.ai_stats_id = self.canvas.create_text(WIDTH / 2, 90, text="", fill=TEXT_COLOR,
                                                   font=("Helvetica", 10, "bold"))
        self.score_id = self.canvas.create_text(WIDTH / 2, 60, text="0", fill=TEXT_COLOR,
                                                font=("Helvetica", 28, "bold"))
        self.message_id = self.canvas.create_text(WIDTH / 2, HEIGHT / 2 - 40,
                                                  text="ESPACO para comecar",
                                                  fill=TEXT_COLOR, font=("Helvetica", 16, "bold"))
        self.submessage_id = self.canvas.create_text(WIDTH / 2, HEIGHT / 2,
                                                     text="ESPACO/UP: pular | R: reiniciar | A: IA | L: aprender | +/-: velocidade | T: turbo",
                                                     fill=TEXT_COLOR, font=("Helvetica", 12))
        self._update_ai_status()
        self._init_stats_window()

        # pool de canos
        self.pipe_pool: List[PipePair] = []
        self.active_pipes: List[PipePair] = []
        self._init_pipes(pool_size=6)

        # estado
        self.state = "ready"
        self.time_since_pipe = 0.0

        # loop
        self.last_time = time.perf_counter()
        self.accumulator = 0.0
        self.frame_ms = int(1000 / FPS)

        # input
        self.root.bind("<space>", self.on_space)
        self.root.bind("<Up>", self.on_space)
        self.root.bind("r", self.on_restart)
        self.root.bind("R", self.on_restart)
        self.root.bind("a", self.on_toggle_ai)
        self.root.bind("A", self.on_toggle_ai)
        self.root.bind("l", self.on_toggle_learn)
        self.root.bind("L", self.on_toggle_learn)
        self.root.bind("<Key-plus>", self.on_speed_up)
        self.root.bind("<Key-equal>", self.on_speed_up)
        self.root.bind("<Key-minus>", self.on_speed_down)
        self.root.bind("<Key-underscore>", self.on_speed_down)
        self.root.bind("t", self.on_toggle_turbo)
        self.root.bind("T", self.on_toggle_turbo)
        self.root.bind("s", self.on_toggle_stable)
        self.root.bind("S", self.on_toggle_stable)

    def _init_pipes(self, pool_size: int) -> None:
        for _ in range(pool_size):
            top_id = self.canvas.create_rectangle(0, 0, 0, 0, fill=PIPE_COLOR, outline="")
            bottom_id = self.canvas.create_rectangle(0, 0, 0, 0, fill=PIPE_COLOR, outline="")
            pipe = PipePair(top_id=top_id, bottom_id=bottom_id)
            self._hide_pipe(pipe)
            self.pipe_pool.append(pipe)

    def _hide_pipe(self, pipe: PipePair) -> None:
        self.canvas.coords(pipe.top_id, -100, -100, -50, -50)
        self.canvas.coords(pipe.bottom_id, -100, -100, -50, -50)

    def _update_pipe_draw(self, pipe: PipePair) -> None:
        top_height = pipe.gap_y - PIPE_GAP / 2
        bottom_y = pipe.gap_y + PIPE_GAP / 2
        self.canvas.coords(pipe.top_id,
                           pipe.x, 0,
                           pipe.x + PIPE_WIDTH, top_height)
        self.canvas.coords(pipe.bottom_id,
                           pipe.x, bottom_y,
                           pipe.x + PIPE_WIDTH, HEIGHT - FLOOR_HEIGHT)

    def _update_bird_draw(self) -> None:
        self.canvas.coords(self.bird_id,
                           self.bird.x - self.bird.r,
                           self.bird.y - self.bird.r,
                           self.bird.x + self.bird.r,
                           self.bird.y + self.bird.r)

    def _update_ai_vision(self) -> None:
        if not self.ai_enabled:
            self.canvas.coords(self.ai_line_id, -100, -100, -50, -50)
            return
        next_pipe = get_next_pipe(self.active_pipes, self.bird.x)
        if next_pipe is None:
            self.canvas.coords(self.ai_line_id, -100, -100, -50, -50)
            return
        target_x = next_pipe.x + PIPE_WIDTH / 2
        target_y = next_pipe.gap_y
        self.canvas.coords(self.ai_line_id, self.bird.x, self.bird.y, target_x, target_y)

    def _init_stats_window(self) -> None:
        self.stats_window = tk.Toplevel(self.root)
        self.stats_window.title("IA Stats - Enhanced")
        self.stats_window.resizable(False, False)
        self.stats_window.protocol("WM_DELETE_WINDOW", self._on_close_stats)

        self.stats_text = tk.StringVar()
        label = tk.Label(self.stats_window, textvariable=self.stats_text, justify="left")
        label.pack(padx=10, pady=8, anchor="w")

        graph_w = GRAPH_WIDTH + GRAPH_PADDING * 2
        graph_h = GRAPH_HEIGHT + GRAPH_PADDING * 2
        self.graph_canvas = tk.Canvas(self.stats_window, width=graph_w, height=graph_h, highlightthickness=0)
        self.graph_canvas.pack(padx=10, pady=6)

        graph_x = GRAPH_PADDING
        graph_y = GRAPH_PADDING
        self.graph_rect_id = self.graph_canvas.create_rectangle(
            graph_x, graph_y, graph_x + GRAPH_WIDTH, graph_y + GRAPH_HEIGHT,
            outline="#3b3b3b", width=1
        )
        self.graph_line_id = self.graph_canvas.create_line(-100, -100, -50, -50, fill="#2b6db6", width=2)
        self.graph_avg_id = self.graph_canvas.create_line(-100, -100, -50, -50, fill="#7a7a7a", width=1)
        self.graph_text_id = self.graph_canvas.create_text(
            graph_x + GRAPH_WIDTH / 2, graph_y + GRAPH_HEIGHT + 10,
            text="", fill=TEXT_COLOR, font=("Helvetica", 9)
        )

    def _on_close_stats(self) -> None:
        if self.stats_window is not None:
            self.stats_window.destroy()
            self.stats_window = None

    def _update_ai_status(self) -> None:
        label = "IA"
        if self.ai_agent is not None and hasattr(self.ai_agent, "label"):
            label = f"IA {self.ai_agent.label}"
        if not self.ai_enabled:
            status = f"{label}: OFF"
        else:
            if self.ai_train:
                status = f"{label}: ON LRN eps {self.ai_agent.epsilon:.3f}"
            else:
                status = f"{label}: ON"
        self.canvas.itemconfig(self.ai_status_id, text=status)
        stats = ""
        if self.ai_enabled:
            loss = "-" if self.ai_last_loss is None else f"{self.ai_last_loss:.3f}"
            turbo = "ON" if self.auto_turbo else "OFF"
            stable = "ON" if self.auto_stable else "OFF"
            using_best = "-"
            if hasattr(self.ai_agent, "use_best"):
                using_best = "YES" if self.ai_agent.use_best else "NO"
            guide = "-"
            if hasattr(self.ai_agent, "guide_only"):
                guide = "ON" if self.ai_agent.guide_only else "OFF"
            device = "-"
            if hasattr(self.ai_agent, "device_name"):
                device = self.ai_agent.device_name
            dueling = "-"
            if hasattr(self.ai_agent, "use_dueling"):
                dueling = "YES" if self.ai_agent.use_dueling else "NO"
            stats = (f"ep {self.ai_episode} speed x{self.speed_mul:.1f} loss {loss} "
                     f"turbo {turbo} stable {stable} best {using_best} guide {guide} "
                     f"dueling {dueling} dev {device}")
        self.canvas.itemconfig(self.ai_stats_id, text="Veja a janela de stats")
        if getattr(self, "stats_window", None) is not None:
            self.stats_text.set(
                f"{status}\n{stats}\nscore {self.score} best {self.best_score} avg {self._avg_score():.1f} "
                f"fail {self.fail_streak}\nPIPE_GAP {PIPE_GAP:.0f} PIPE_SPEED {PIPE_SPEED:.0f}"
            )
        self._update_graph()

    def _update_graph(self) -> None:
        if getattr(self, "stats_window", None) is None:
            return
        if not self.score_history:
            self.graph_canvas.coords(self.graph_line_id, -100, -100, -50, -50)
            self.graph_canvas.coords(self.graph_avg_id, -100, -100, -50, -50)
            self.graph_canvas.itemconfig(self.graph_text_id, text="")
            return
        graph_x = GRAPH_PADDING
        graph_y = GRAPH_PADDING
        history = self.score_history[-60:]
        max_score = max(1, max(history))
        points = []
        for i, s in enumerate(history):
            x = graph_x + (i / max(1, len(history) - 1)) * GRAPH_WIDTH
            y = graph_y + GRAPH_HEIGHT - (s / max_score) * GRAPH_HEIGHT
            points.extend([x, y])
        if len(points) >= 4:
            self.graph_canvas.coords(self.graph_line_id, *points)

        if len(history) >= 3:
            avg = sum(history) / len(history)
            y_avg = graph_y + GRAPH_HEIGHT - (avg / max_score) * GRAPH_HEIGHT
            self.graph_canvas.coords(self.graph_avg_id, graph_x, y_avg, graph_x + GRAPH_WIDTH, y_avg)
        else:
            self.graph_canvas.coords(self.graph_avg_id, -100, -100, -50, -50)

        avg_display = sum(history) / len(history)
        self.graph_canvas.itemconfig(self.graph_text_id, text=f"best {self.best_score} avg {avg_display:.1f}")

    def _avg_score(self) -> float:
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)

    def on_space(self, _event=None) -> None:
        if self.state == "ready":
            self.start_game()
        elif self.state == "playing":
            self.bird.flap()

    def on_restart(self, _event=None) -> None:
        if self.state == "game_over":
            self.start_game()

    def on_toggle_ai(self, _event=None) -> None:
        if self.ai_agent is None:
            self.canvas.itemconfig(self.submessage_id, text="IA nao carregada. Rode --train.")
            return
        self.ai_enabled = not self.ai_enabled
        if not self.ai_train:
            self.ai_agent.epsilon = 0.0
        self._update_ai_status()
        if self.ai_enabled and self.state != "playing":
            self.start_game()

    def on_toggle_learn(self, _event=None) -> None:
        if self.ai_agent is None:
            self.canvas.itemconfig(self.submessage_id, text="IA nao carregada. Rode --train.")
            return
        self.ai_train = not self.ai_train
        if self.ai_train and not self.ai_enabled:
            self.ai_enabled = True
        if self.ai_train:
            self.ai_agent.epsilon = self.ai_epsilon
            if self.speed_mul < 2.0:
                self.speed_mul = 2.0
        else:
            self.ai_agent.epsilon = 0.0
        self._update_ai_status()
        if self.ai_enabled and self.state != "playing":
            self.start_game()

    def on_speed_up(self, _event=None) -> None:
        self.speed_mul = min(TRAIN_SPEED_MAX, self.speed_mul + 0.5)
        self._update_ai_status()

    def on_speed_down(self, _event=None) -> None:
        self.speed_mul = max(0.5, self.speed_mul - 0.5)
        self._update_ai_status()

    def on_toggle_turbo(self, _event=None) -> None:
        self.auto_turbo = not self.auto_turbo
        self._update_ai_status()

    def on_toggle_stable(self, _event=None) -> None:
        self.auto_stable = not self.auto_stable
        if hasattr(self.ai_agent, "use_best"):
            self.ai_agent.use_best = self.auto_stable
        self._update_ai_status()

    def start_game(self) -> None:
        self.state = "playing"
        self.score = 0
        self.canvas.itemconfig(self.score_id, text=str(self.score))
        self.canvas.itemconfig(self.message_id, text="")
        self.canvas.itemconfig(self.submessage_id, text="")
        self.time_since_pipe = 0.0
        self.ai_decision_timer = 0.0
        self.ai_last_obs = None
        self.ai_last_action = 0
        self.ai_reward_accum = 0.0
        self.ai_last_loss = None
        self.history_buffer.clear()

        if self.ai_enabled and self.ai_train and self.ai_agent is not None:
            if not hasattr(self.ai_agent, "decay_epsilon"):
                if self.ai_episode > 0:
                    self.ai_epsilon = max(AI_EPS_MIN, self.ai_epsilon * AI_EPS_DECAY)
                self.ai_agent.epsilon = self.ai_epsilon
            self.ai_episode += 1
            if hasattr(self.ai_agent, "guide_only"):
                if self.force_guide_episodes > 0:
                    self.ai_agent.guide_only = True
                    self.force_guide_episodes -= 1
                else:
                    self.ai_agent.guide_only = self.ai_episode <= NN_WARMUP_EPISODES
            self._update_ai_status()

        self.bird.reset(y=HEIGHT * 0.45)
        self._update_bird_draw()

        for pipe in self.active_pipes:
            pipe.deactivate()
            self._hide_pipe(pipe)
            self.pipe_pool.append(pipe)
        self.active_pipes.clear()

        if self.ai_enabled:
            if self.ai_train:
                assist_prob = assist_prob_for_episode(self.ai_episode)
                gap_y = sample_gap_y(self.bird.y, assist_prob)
            else:
                gap_y = None
            self.spawn_pipe(x=WIDTH * 0.6, gap_y=gap_y)

    def game_over(self) -> None:
        self.state = "game_over"
        self.canvas.itemconfig(self.message_id, text="Fim de jogo")
        self.canvas.itemconfig(self.submessage_id, text="Pressione R para recomecar")
        
        if self.ai_train and self.ai_agent is not None:
            if self.ai_episode > 0 and (self.ai_episode % self.ai_save_every) == 0:
                if hasattr(self.ai_agent, "net"):
                    save_dqn(NN_MODEL_PATH, self.ai_agent)
                    print(f"Modelo salvo em {NN_MODEL_PATH} (episodio {self.ai_episode})")
                elif hasattr(self.ai_agent, "qtable"):
                    save_qtable(AI_QTABLE_PATH, self.ai_agent.qtable)
            
            self.score_history.append(self.score)
            if self.score > self.best_score:
                self.best_score = self.score
            if hasattr(self.ai_agent, "update_best"):
                self.ai_agent.update_best(self.score)
            if hasattr(self.ai_agent, "decay_epsilon"):
                self.ai_agent.decay_epsilon(success=self.score >= STABLE_SCORE_ON, failed=self.score <= 0)
            if hasattr(self.ai_agent, "epsilon") and len(self.score_history) >= 10:
                recent_avg = sum(self.score_history[-10:]) / 10.0
                if recent_avg < 1.0 and self.ai_agent.epsilon <= NN_EPS_MIN + 1e-6:
                    self.ai_agent.epsilon = max(self.ai_agent.epsilon, NN_EPS_REHEAT)
            if self.score <= 0:
                self.fail_streak += 1
            else:
                self.fail_streak = 0
            if self.fail_streak >= NN_FAIL_GUIDE_TRIGGER:
                if (self.fail_streak % NN_FAIL_GUIDE_TRIGGER) == 0:
                    self.force_guide_episodes = NN_FAIL_GUIDE_EPISODES
                    if hasattr(self.ai_agent, "use_best"):
                        self.ai_agent.use_best = False
            if self.auto_turbo:
                self._auto_turbo()
            self._auto_stabilize()
            self._update_graph()
            self._update_ai_status()
        
        if self.ai_enabled:
            self.root.after(350, self.start_game)

    def spawn_pipe(self, x: Optional[float] = None, gap_y: Optional[int] = None) -> None:
        if not self.pipe_pool:
            return
        if x is None:
            x = WIDTH + PIPE_WIDTH
        if gap_y is None:
            assist_prob = 0.0
            if self.ai_train:
                assist_prob = assist_prob_for_episode(self.ai_episode)
            gap_y = sample_gap_y(self.bird.y, assist_prob)
        pipe = self.pipe_pool.pop()
        pipe.reset(x=x, gap_y=gap_y)
        self.active_pipes.append(pipe)
        self._update_pipe_draw(pipe)

    def _auto_turbo(self) -> None:
        if len(self.score_history) < 6:
            return
        recent = self.score_history[-6:]
        avg = sum(recent) / len(recent)
        if avg >= 6 and self.speed_mul < TRAIN_SPEED_MAX:
            self.speed_mul = min(TRAIN_SPEED_MAX, self.speed_mul + 0.5)
        elif avg < 3 and self.speed_mul > 1.0:
            self.speed_mul = max(1.0, self.speed_mul - 0.5)

    def _auto_stabilize(self) -> None:
        if not self.auto_stable or not hasattr(self.ai_agent, "use_best"):
            return
        if len(self.score_history) < STABLE_MIN_HISTORY:
            return
        recent = self.score_history[-STABLE_MIN_HISTORY:]
        avg = sum(recent) / len(recent)
        if avg >= STABLE_SCORE_ON:
            self.ai_agent.use_best = True
        elif avg <= STABLE_SCORE_OFF:
            self.ai_agent.use_best = False

    def _get_ai_obs(self) -> AIObs:
        use_enhanced = isinstance(self.ai_agent, DoubleDQNAgent) or (
            hasattr(self.ai_agent, "layers") and self.ai_agent.layers[0] == 37
        )
        return make_obs(self.bird.x, self.bird.y, self.bird.vy, self.active_pipes, 
                       self.history_buffer, use_enhanced)

    def update_logic(self, dt: float) -> None:
        did_flap = False
        if self.ai_enabled and self.ai_agent is not None:
            self.ai_decision_timer += dt
            if self.ai_decision_timer >= self.ai_decision_dt:
                self.ai_decision_timer -= self.ai_decision_dt
                obs = self._get_ai_obs()
                
                # Atualizar histórico
                self.history_buffer.append(obs)
                
                if self.ai_train and self.ai_last_obs is not None:
                    loss = self.ai_agent.learn(self.ai_last_obs, self.ai_last_action,
                                               self.ai_reward_accum, obs, False)
                    if loss is not None:
                        self.ai_last_loss = loss
                        self._update_ai_status()
                    self.ai_reward_accum = 0.0
                action = self.ai_agent.act(obs)
                self.ai_last_obs = obs
                self.ai_last_action = action
                if action == 1:
                    t_apex = max(0.0, -JUMP_VELOCITY / GRAVITY)
                    y_apex = predict_y(self.bird.y, JUMP_VELOCITY, t_apex)
                    if (self.bird.y - BIRD_RADIUS) < 16 or (y_apex - BIRD_RADIUS) < 8:
                        action = 0
                if action == 1:
                    self.bird.flap()
                    did_flap = True

        self.bird.update(dt)
        self._update_bird_draw()

        self.time_since_pipe += dt
        if self.time_since_pipe >= PIPE_INTERVAL:
            self.time_since_pipe -= PIPE_INTERVAL
            self.spawn_pipe()

        scored = False
        for pipe in list(self.active_pipes):
            pipe.update(dt)
            self._update_pipe_draw(pipe)

            if not pipe.scored and (self.bird.x > pipe.x + PIPE_WIDTH):
                pipe.scored = True
                self.score += 1
                self.canvas.itemconfig(self.score_id, text=str(self.score))
                scored = True

            if pipe.offscreen():
                pipe.deactivate()
                self._hide_pipe(pipe)
                self.active_pipes.remove(pipe)
                self.pipe_pool.append(pipe)

        self._update_ai_vision()

        dead = self.check_collisions()
        if self.ai_train and self.ai_agent is not None and self.ai_last_obs is not None:
            obs_now = self._get_ai_obs()
            next_pipe = get_next_pipe(self.active_pipes, self.bird.x)
            reward = enhanced_reward(self.bird.y, self.bird.vy, obs_now.dy, 
                                   1 if did_flap else 0, scored, dead, next_pipe)
            self.ai_reward_accum += reward
            if dead:
                loss = self.ai_agent.learn(self.ai_last_obs, self.ai_last_action,
                                           self.ai_reward_accum, self.ai_last_obs, True)
                if loss is not None:
                    self.ai_last_loss = loss
                    self._update_ai_status()
                self.ai_reward_accum = 0.0

        if dead:
            self.game_over()

    def check_collisions(self) -> bool:
        if (self.bird.y + self.bird.r) >= (HEIGHT - FLOOR_HEIGHT):
            return True
        if (self.bird.y - self.bird.r) <= 0:
            return True

        for pipe in self.active_pipes:
            top_height = pipe.gap_y - PIPE_GAP / 2
            bottom_y = pipe.gap_y + PIPE_GAP / 2

            if circle_rect_collision(self.bird.x, self.bird.y, self.bird.r,
                                     pipe.x, 0, PIPE_WIDTH, top_height):
                return True

            if circle_rect_collision(self.bird.x, self.bird.y, self.bird.r,
                                     pipe.x, bottom_y, PIPE_WIDTH, HEIGHT - FLOOR_HEIGHT - bottom_y):
                return True

        return False

    def loop(self) -> None:
        now = time.perf_counter()
        frame_dt = now - self.last_time
        self.last_time = now

        frame_dt = min(frame_dt, 0.05)
        self.accumulator += frame_dt * self.speed_mul

        while self.accumulator >= FIXED_DT:
            if self.state == "playing":
                self.update_logic(FIXED_DT)
            self.accumulator -= FIXED_DT

        self.root.after(self.frame_ms, self.loop)

    def run(self) -> None:
        self.loop()
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Flappy Bird com IA aprimorada.")
    parser.add_argument("--train", action="store_true", help="Treina a IA em modo rapido (sem janela).")
    parser.add_argument("--play-ai", action="store_true", help="Roda o jogo com a IA.")
    parser.add_argument("--learn", action="store_true", help="Roda o jogo com a IA aprendendo ao vivo.")
    parser.add_argument("--nn-train", action="store_true", help="Treina a IA com rede neural (DQN aprimorada).")
    parser.add_argument("--nn-play", action="store_true", help="Roda o jogo com a IA de rede neural.")
    parser.add_argument("--nn-learn", action="store_true", help="Treina a DQN com janela (ao vivo).")
    parser.add_argument("--nn-heavy", action="store_true", help="Usa rede maior e replay maior (mais CPU/RAM).")
    parser.add_argument("--nn-backend", choices=["auto", "numpy", "torch"], default="auto",
                        help="Backend da DQN (auto|numpy|torch). Torch usa CUDA se disponivel.")
    parser.add_argument("--use-enhanced", action="store_true", default=True,
                        help="Usa versão aprimorada (Double DQN + Prioritized Replay + Dueling).")
    parser.add_argument("--speed", type=float, default=None, help="Multiplicador de velocidade da simulacao.")
    parser.add_argument("--episodes", type=int, default=1200, help="Numero de episodios de treino.")
    parser.add_argument("--max-steps", type=int, default=20000, help="Limite de passos por episodio.")
    parser.add_argument("--seed", type=int, default=None, help="Semente para reproducibilidade.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        if np is not None:
            np.random.seed(args.seed)

    qtable = None
    nn_agent = None

    if args.nn_train or args.nn_play or args.nn_learn:
        use_cuda = args.nn_backend in ("torch", "auto") and torch is not None and torch.cuda.is_available()
        configure_nn(args.nn_heavy, use_cuda=use_cuda)

    if args.nn_backend == "torch" and torch is None:
        print("Torch nao instalado. Rode: pip install torch")
        return

    if args.nn_train:
        print("=" * 60)
        print("Treinando IA (DQN Aprimorada)")
        print("=" * 60)
        print("Melhorias implementadas:")
        print("  ✓ Prioritized Experience Replay")
        print("  ✓ Double DQN")
        print("  ✓ Dueling DQN Architecture")
        print("  ✓ Features aprimoradas (37 features)")
        print("  ✓ Sistema de recompensas denso")
        print("  ✓ Curriculum learning adaptativo")
        print("  ✓ Histórico de frames")
        print("=" * 60)
        nn_agent = train_dqn(args.episodes, args.max_steps, backend=args.nn_backend, use_enhanced=args.use_enhanced)
        save_dqn(NN_MODEL_PATH, nn_agent)
        print(f"\nIA salva em {NN_MODEL_PATH}")
        if not args.nn_play:
            return

    if args.nn_play or args.nn_learn:
        if nn_agent is None:
            try:
                nn_agent = load_dqn(NN_MODEL_PATH, backend=args.nn_backend, use_enhanced=args.use_enhanced)
                print(f"Modelo carregado de {NN_MODEL_PATH}")
            except FileNotFoundError:
                if args.nn_learn:
                    use_torch = args.nn_backend in ("torch", "auto") and torch is not None
                    if args.use_enhanced and not use_torch:
                        print("Criando novo DoubleDQNAgent com Dueling DQN")
                        nn_agent = DoubleDQNAgent(NN_LAYERS, use_dueling=True)
                    elif use_torch:
                        device = get_torch_device(prefer_cuda=True)
                        nn_agent = TorchDQNAgent(NN_LAYERS, device)
                    else:
                        require_numpy()
                        nn_agent = DQNAgent(NN_LAYERS)
                else:
                    print("Modelo DQN nao encontrado. Rode --nn-train primeiro.")
                    return
        game = FlappyGame(ai_agent=nn_agent)
        if args.nn_learn:
            game.ai_train = True
            game.ai_enabled = True
            game.ai_agent.epsilon = game.ai_agent.epsilon
            if game.speed_mul < 2.0:
                game.speed_mul = 2.0
        if args.speed is not None:
            game.speed_mul = clamp(args.speed, 0.5, TRAIN_SPEED_MAX)
        game.start_game()
        game.run()
        return

    if args.train:
        print("Treinando IA (Q-learning)...")
        qtable = train_qtable(args.episodes, args.max_steps)
        save_qtable(AI_QTABLE_PATH, qtable)
        print(f"IA salva em {AI_QTABLE_PATH}")

    if args.play_ai or args.learn:
        if qtable is None:
            try:
                qtable = load_qtable(AI_QTABLE_PATH)
            except FileNotFoundError:
                qtable = {}
        agent = QAgent(qtable, epsilon=0.0)
        game = FlappyGame(ai_agent=agent)
        if args.learn:
            game.ai_train = True
            game.ai_enabled = True
            game.ai_agent.epsilon = game.ai_epsilon
        game.start_game()
        game.run()
        return

    if not args.train:
        FlappyGame().run()


if __name__ == "__main__":
    main()
