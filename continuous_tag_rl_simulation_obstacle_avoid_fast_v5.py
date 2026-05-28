import os
import pickle
import random
import math
from collections import defaultdict

# =========================================================
# 정사각형 경기장 술래잡기 강화학습 시뮬레이션 v2
# - 격자칸 없음
# - 연속 좌표 기반 자유 이동
# - Q-learning 기반 술래 AI / 도망자 AI
# - 학습 데이터 저장/불러오기
# - 학습 과정 GUI 시각화
# - 도망자 생존 시간을 "분" 단위로 설정
# - 학습 시각화 배속 조절 가능
# - 선분 벽 장애물 추가/삭제 가능
# - 벽이 술래와 도망자 사이를 막으면 잡힘 판정 차단
# - AI 상태에 8방향 벽 감지 정보를 포함하여 벽 회피 학습 강화
# - 벽 감지 캐시로 fast 학습 속도 개선
# =========================================================

# -----------------------------
# 경기장 설정
# -----------------------------
ARENA_SIZE = 500.0
CAPTURE_RADIUS = 18.0

# 한 번의 내부 시뮬레이션 step이 의미하는 실제 게임 시간
# 1.0이면 1 step = 1초
SIM_SECONDS_PER_STEP = 1.0

# 도망자가 버텨야 하는 시간
# 예: 1.0분 = 60초 = 60 step
SURVIVE_MINUTES = 1.0

# 속도 설정
# 단위: 1 step마다 이동하는 거리
CHASER_SPEED = 10.0
RUNNER_SPEED = 7.0

# 시작 시 너무 가까이 붙어 있지 않도록 하는 최소 거리
MIN_START_DISTANCE = 150.0

# 시작 위치 모드
# "random": 매 판마다 술래와 도망자의 시작 위치를 랜덤으로 설정
# "fixed" : 매 판마다 술래와 도망자의 시작 위치를 고정 좌표로 설정
START_MODE = "random"

# 고정 시작 위치
FIXED_CHASER_POS = (100.0, 250.0)
FIXED_RUNNER_POS = (400.0, 250.0)

# -----------------------------
# 장애물 벽 설정
# -----------------------------
# 벽은 시작 좌표와 끝 좌표를 가진 선분으로 설정한다.
# 실제 판정에서는 DEFAULT_WALL_THICKNESS만큼 두께가 있는 벽처럼 처리한다.
DEFAULT_WALL_THICKNESS = 14.0
WALLS = []

# 벽 회피 학습 설정
# AI가 현재 위치에서 각 방향으로 이만큼 앞을 미리 검사한다.
# 값이 클수록 벽을 더 일찍 감지한다.
WALL_SENSOR_LOOKAHEAD = 2.0

# 벽에 부딪혔을 때의 벌점
# 기존 -0.10은 너무 약해서 벽에 계속 박는 행동을 쉽게 버리지 못했다.
WALL_HIT_PENALTY = 3.0

# 벽에 부딪혀 실제 위치가 거의 변하지 않았을 때 추가 벌점
STUCK_PENALTY = 2.0

# fast 학습 최적화 설정
# 벽 감지 결과를 위치별로 저장해 재사용한다.
# 값이 작을수록 정밀하지만 느리고, 값이 클수록 빠르지만 대략적으로 감지한다.
WALL_SENSOR_CACHE_GRID = 15.0
WALL_SENSOR_CACHE = {}

# -----------------------------
# 학습 시각화 설정
# -----------------------------
# 1배속이면 실제 시간과 비슷하게 1 step을 1초 간격으로 보여준다.
# 20배속이면 1 step을 약 0.05초 간격으로 보여준다.
TRAIN_SPEED_MULTIPLIER = 20.0

# -----------------------------
# Q-learning 설정
# -----------------------------
ALPHA = 0.20
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 20000

# 상태 구간화 설정
# 실제 이동은 연속 좌표지만, Q-table에는 압축된 상태만 저장한다.
ANGLE_BINS = 8
DISTANCE_BINS = 8
WALL_MARGIN = 80.0

# 저장 파일 이름
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
SAVE_FILE = os.path.join(BASE_DIR, "continuous_tag_ai_learning_data_obstacle_avoid.pkl")

# -----------------------------
# 행동 설정
# 8방향 자유 이동
# -----------------------------
RAW_ACTIONS = [
    (0.0, 1.0),    # UP
    (0.0, -1.0),   # DOWN
    (-1.0, 0.0),   # LEFT
    (1.0, 0.0),    # RIGHT
    (1.0, 1.0),    # UP_RIGHT
    (-1.0, 1.0),   # UP_LEFT
    (1.0, -1.0),   # DOWN_RIGHT
    (-1.0, -1.0),  # DOWN_LEFT
]

ACTION_NAMES = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "UP_RIGHT",
    "UP_LEFT",
    "DOWN_RIGHT",
    "DOWN_LEFT",
]

# 대각선 이동이 더 빨라지지 않도록 방향 벡터를 정규화한다.
ACTIONS = []
for dx, dy in RAW_ACTIONS:
    length = math.hypot(dx, dy)
    ACTIONS.append((dx / length, dy / length))


class QLearningAgent:
    def __init__(self, name):
        self.name = name
        self.q_table = defaultdict(lambda: [0.0 for _ in ACTIONS])
        self.score = 0

    def choose_action(self, state, epsilon):
        """
        epsilon-greedy 방식으로 행동 선택

        epsilon 확률       : 랜덤 행동
        1 - epsilon 확률   : 현재 Q값이 가장 높은 행동
        """
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))

        q_values = self.q_table[state]
        max_q = max(q_values)

        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        """
        Q-learning 업데이트 공식

        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        target = reward + GAMMA * next_max_q
        self.q_table[state][action] += ALPHA * (target - current_q)

    def clear_learning_data(self):
        self.q_table.clear()
        self.score = 0


def get_max_steps():
    """
    도망자가 버텨야 하는 시간을 step 수로 변환한다.

    예:
    SURVIVE_MINUTES = 1.0
    SIM_SECONDS_PER_STEP = 1.0
    => 60 step
    """
    survive_seconds = SURVIVE_MINUTES * 60.0
    return max(1, int(math.ceil(survive_seconds / SIM_SECONDS_PER_STEP)))


def get_survive_seconds():
    return SURVIVE_MINUTES * 60.0


def get_animation_delay_ms():
    """
    GUI에서 다음 step으로 넘어가기까지의 시간.

    1배속:
        1 step = SIM_SECONDS_PER_STEP초에 해당하므로
        화면에서도 실제 시간에 가깝게 보여준다.

    20배속:
        20배 빠르게 보여준다.
    """
    speed = max(0.1, TRAIN_SPEED_MULTIPLIER)
    delay = int(1000 * SIM_SECONDS_PER_STEP / speed)
    return max(1, delay)


def clamp(value, low, high):
    return max(low, min(high, value))


def distance(pos1, pos2):
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])


def dot(ax, ay, bx, by):
    return ax * bx + ay * by


def distance_point_to_segment(point, seg_start, seg_end):
    """
    점과 선분 사이의 최단 거리.
    벽 충돌 및 시야 차단 판정에 사용한다.
    """
    px, py = point
    ax, ay = seg_start
    bx, by = seg_end

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby

    if ab_len_sq == 0:
        return distance(point, seg_start)

    t = dot(apx, apy, abx, aby) / ab_len_sq
    t = clamp(t, 0.0, 1.0)

    closest = (ax + abx * t, ay + aby * t)
    return distance(point, closest)


def orientation(a, b, c):
    """
    세 점의 방향성 판단.
    값이 0에 가까우면 거의 일직선이다.
    """
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

    if abs(value) < 1e-9:
        return 0

    return 1 if value > 0 else 2


def on_segment(a, b, c):
    """b가 선분 ac 위에 있는지 확인한다."""
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(p1, p2, q1, q2):
    """두 선분이 서로 교차하는지 확인한다."""
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def distance_segment_to_segment(p1, p2, q1, q2):
    """
    두 선분 사이의 최단 거리.
    선분이 교차하면 0이다.
    """
    if segments_intersect(p1, p2, q1, q2):
        return 0.0

    return min(
        distance_point_to_segment(p1, q1, q2),
        distance_point_to_segment(p2, q1, q2),
        distance_point_to_segment(q1, p1, p2),
        distance_point_to_segment(q2, p1, p2),
    )


def make_wall(x1, y1, x2, y2, thickness=None):
    """벽 정보를 dict 형태로 만든다."""
    if thickness is None:
        thickness = DEFAULT_WALL_THICKNESS

    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "thickness": float(thickness),
    }


def wall_points(wall):
    return (wall["x1"], wall["y1"]), (wall["x2"], wall["y2"])


def clamp_wall_to_arena(wall):
    """
    벽의 시작점과 끝점이 경기장 밖으로 나가지 않도록 보정한다.
    """
    wall["x1"] = clamp(wall["x1"], 0.0, ARENA_SIZE)
    wall["y1"] = clamp(wall["y1"], 0.0, ARENA_SIZE)
    wall["x2"] = clamp(wall["x2"], 0.0, ARENA_SIZE)
    wall["y2"] = clamp(wall["y2"], 0.0, ARENA_SIZE)
    wall["thickness"] = max(1.0, wall["thickness"])
    return wall


def is_point_inside_wall(point, wall, extra_margin=0.0):
    """
    점이 벽 두께 안에 들어와 있는지 확인한다.
    """
    a, b = wall_points(wall)
    return distance_point_to_segment(point, a, b) <= wall["thickness"] / 2.0 + extra_margin


def is_blocked_by_any_wall(path_start, path_end, extra_margin=0.0):
    """
    두 점 사이의 선분이 어떤 벽과 닿거나 통과하는지 확인한다.

    - 이동 경로가 벽에 닿으면 이동을 막는다.
    - 술래와 도망자 사이의 시야가 벽에 막혔는지도 이 함수로 판정한다.
    """
    if not WALLS:
        return False

    if path_start == path_end:
        return any(is_point_inside_wall(path_start, wall, extra_margin) for wall in WALLS)

    for wall in WALLS:
        wall_start, wall_end = wall_points(wall)
        d = distance_segment_to_segment(path_start, path_end, wall_start, wall_end)

        if d <= wall["thickness"] / 2.0 + extra_margin:
            return True

    return False


def can_capture(chaser_pos, runner_pos):
    """
    술래가 도망자를 잡을 수 있는지 확인한다.

    조건:
    1. 두 AI의 거리가 CAPTURE_RADIUS 이하여야 한다.
    2. 술래와 도망자 사이의 직선 경로가 벽에 막혀 있지 않아야 한다.
    """
    if distance(chaser_pos, runner_pos) > CAPTURE_RADIUS:
        return False

    # 둘 사이에 벽이 있으면 잡지 못한다.
    if is_blocked_by_any_wall(chaser_pos, runner_pos):
        return False

    return True


def clear_wall_sensor_cache():
    """벽 배치가 바뀌면 벽 감지 캐시를 비운다."""
    WALL_SENSOR_CACHE.clear()


def add_wall(x1, y1, x2, y2, thickness=None):
    """벽을 추가한다."""
    wall = make_wall(x1, y1, x2, y2, thickness)
    wall = clamp_wall_to_arena(wall)

    if distance((wall["x1"], wall["y1"]), (wall["x2"], wall["y2"])) < 1.0:
        print("벽의 시작점과 끝점이 너무 가깝다. 벽을 추가하지 않았다.")
        return False

    WALLS.append(wall)
    clear_wall_sensor_cache()
    print(
        f"벽 추가 완료: "
        f"({wall['x1']:.1f}, {wall['y1']:.1f}) -> "
        f"({wall['x2']:.1f}, {wall['y2']:.1f}), "
        f"두께 {wall['thickness']:.1f}"
    )
    return True


def list_walls():
    """현재 등록된 벽 목록을 출력한다."""
    if not WALLS:
        print("현재 등록된 벽이 없다.")
        return

    print("\n===== 벽 목록 =====")
    for i, wall in enumerate(WALLS, start=1):
        print(
            f"{i}. "
            f"({wall['x1']:.1f}, {wall['y1']:.1f}) -> "
            f"({wall['x2']:.1f}, {wall['y2']:.1f}), "
            f"두께 {wall['thickness']:.1f}"
        )


def clear_walls():
    """모든 벽을 삭제한다."""
    WALLS.clear()
    clear_wall_sensor_cache()
    print("모든 벽을 삭제했다.")
    print("이 명령 후 자동 저장되므로 다음 실행 때도 벽 0개 상태가 유지된다.")


def remove_wall(index):
    """번호로 벽 하나를 삭제한다."""
    if index < 1 or index > len(WALLS):
        print("삭제할 벽 번호가 올바르지 않다.")
        return False

    removed = WALLS.pop(index - 1)
    clear_wall_sensor_cache()
    print(
        f"벽 삭제 완료: "
        f"({removed['x1']:.1f}, {removed['y1']:.1f}) -> "
        f"({removed['x2']:.1f}, {removed['y2']:.1f})"
    )
    return True


def get_wall_sensor_cache_key(position, speed):
    """
    연속 좌표를 캐시용 격자 좌표로 압축한다.

    주의:
    실제 게임 이동은 여전히 연속 좌표다.
    여기서는 '벽 감지 상태' 계산만 빠르게 하기 위해 근처 위치를 같은 상태로 묶는다.
    """
    qx = int(position[0] // WALL_SENSOR_CACHE_GRID)
    qy = int(position[1] // WALL_SENSOR_CACHE_GRID)
    return qx, qy, round(speed, 3), len(WALLS)


def get_blocked_action_mask(position, speed):
    """
    현재 위치에서 8방향 중 어느 방향이 벽 또는 경기장 외곽으로 막혀 있는지 반환한다.

    최적화:
    같은 위치 근처에서의 벽 감지 결과는 WALL_SENSOR_CACHE에 저장한다.
    fast 학습에서 같은 주변 위치가 반복해서 등장하므로 계산량이 크게 줄어든다.
    """
    key = get_wall_sensor_cache_key(position, speed)

    if key in WALL_SENSOR_CACHE:
        return WALL_SENSOR_CACHE[key]

    blocked = []

    for dx, dy in ACTIONS:
        x, y = position
        raw_x = x + dx * speed * WALL_SENSOR_LOOKAHEAD
        raw_y = y + dy * speed * WALL_SENSOR_LOOKAHEAD

        outside_arena = (
            raw_x < 0.0
            or raw_x > ARENA_SIZE
            or raw_y < 0.0
            or raw_y > ARENA_SIZE
        )

        if outside_arena:
            blocked.append(1)
            continue

        candidate = (raw_x, raw_y)
        blocked_by_wall = is_blocked_by_any_wall(position, candidate)

        blocked.append(1 if blocked_by_wall else 0)

    result = tuple(blocked)
    WALL_SENSOR_CACHE[key] = result
    return result

def move_position(position, action_index, speed):
    """
    현재 위치에서 선택한 방향으로 이동한다.

    반환값:
    - new_position: 이동 후 좌표
    - hit_wall: 경기장 외벽 또는 장애물 벽에 부딪혔는지 여부
    """
    x, y = position
    dx, dy = ACTIONS[action_index]

    raw_x = x + dx * speed
    raw_y = y + dy * speed

    new_x = clamp(raw_x, 0.0, ARENA_SIZE)
    new_y = clamp(raw_y, 0.0, ARENA_SIZE)

    hit_arena_wall = (new_x != raw_x) or (new_y != raw_y)
    candidate_pos = (new_x, new_y)

    # 이동 경로가 장애물 벽을 통과하면 이동하지 못하게 한다.
    # 단순히 새 위치만 검사하는 것이 아니라, 이전 위치와 새 위치 사이의 경로를 검사한다.
    hit_obstacle_wall = is_blocked_by_any_wall(position, candidate_pos)

    if hit_obstacle_wall:
        return position, True

    return candidate_pos, hit_arena_wall

def random_start_positions():
    """술래와 도망자의 시작 위치를 랜덤으로 생성한다."""
    while True:
        chaser_pos = (
            random.uniform(0.0, ARENA_SIZE),
            random.uniform(0.0, ARENA_SIZE),
        )
        runner_pos = (
            random.uniform(0.0, ARENA_SIZE),
            random.uniform(0.0, ARENA_SIZE),
        )

        if (
            distance(chaser_pos, runner_pos) >= MIN_START_DISTANCE
            and not any(is_point_inside_wall(chaser_pos, wall, extra_margin=5.0) for wall in WALLS)
            and not any(is_point_inside_wall(runner_pos, wall, extra_margin=5.0) for wall in WALLS)
        ):
            return chaser_pos, runner_pos


def get_start_positions():
    """
    현재 START_MODE 설정에 따라 시작 위치를 결정한다.

    START_MODE = "random"
        매 판마다 랜덤 위치에서 시작한다.

    START_MODE = "fixed"
        매 판마다 FIXED_CHASER_POS, FIXED_RUNNER_POS에서 시작한다.
    """
    if START_MODE == "fixed":
        return FIXED_CHASER_POS, FIXED_RUNNER_POS

    return random_start_positions()


def angle_to_bin(dx, dy):
    """상대가 어느 방향에 있는지 ANGLE_BINS개 구간으로 나눈다."""
    angle = math.atan2(dy, dx)
    normalized = (angle + math.pi) / (2 * math.pi)
    return int(normalized * ANGLE_BINS) % ANGLE_BINS


def value_to_bin(value, low, high, bins):
    """연속값을 정해진 개수의 구간으로 나눈다."""
    value = clamp(value, low, high)
    ratio = (value - low) / (high - low)
    index = int(ratio * bins)

    if index >= bins:
        index = bins - 1

    return index


def wall_zone(value):
    """
    현재 위치가 벽 근처인지 판단한다.

    0: 왼쪽 또는 아래쪽 벽 근처
    1: 중앙 영역
    2: 오른쪽 또는 위쪽 벽 근처
    """
    if value < WALL_MARGIN:
        return 0
    if value > ARENA_SIZE - WALL_MARGIN:
        return 2
    return 1


def make_agent_state(self_pos, opponent_pos, self_speed):
    """
    AI가 보는 상태를 만든다.

    실제 좌표는 연속값이지만, Q-table에는 아래 정보만 저장한다.

    상태 = (
        상대가 있는 방향 구간,
        상대와의 거리 구간,
        내 x 위치가 벽 근처인지,
        내 y 위치가 벽 근처인지,
        8방향 벽 감지 정보
    )

    핵심 변경:
    기존에는 AI가 '상대의 방향/거리'만 알고 벽의 위치를 몰랐다.
    그래서 상대를 향해 가다가 벽에 박혀 멈추는 행동을 반복할 수 있었다.

    이제는 현재 위치에서 8방향 중 어느 방향이 막혀 있는지 상태에 포함한다.
    """
    dx = opponent_pos[0] - self_pos[0]
    dy = opponent_pos[1] - self_pos[1]

    angle_bin = angle_to_bin(dx, dy)
    dist_bin = value_to_bin(
        distance(self_pos, opponent_pos),
        0.0,
        math.sqrt(2) * ARENA_SIZE,
        DISTANCE_BINS,
    )

    x_wall = wall_zone(self_pos[0])
    y_wall = wall_zone(self_pos[1])

    blocked_actions = get_blocked_action_mask(self_pos, self_speed)

    return angle_bin, dist_bin, x_wall, y_wall, blocked_actions

def get_epsilon(total_episodes_done):
    """
    누적 학습 게임 수에 따라 epsilon을 줄인다.
    학습 초반에는 탐험을 많이 하고, 후반에는 학습된 행동을 많이 사용한다.
    """
    progress = min(total_episodes_done / EPSILON_DECAY_EPISODES, 1.0)
    return EPSILON_START * (1 - progress) + EPSILON_END * progress


def simulate_one_step(
    chaser_ai,
    runner_ai,
    chaser_pos,
    runner_pos,
    epsilon,
    step_index,
    max_steps,
    training=True,
):
    """
    게임의 한 step만 진행한다.

    이 함수를 사용하면:
    - 콘솔 학습
    - GUI 학습
    - 텍스트 시연
    - GUI 시연

    모두 같은 로직을 공유할 수 있다.
    """
    old_chaser_pos = chaser_pos
    old_runner_pos = runner_pos
    old_distance = distance(chaser_pos, runner_pos)

    chaser_state = make_agent_state(chaser_pos, runner_pos, CHASER_SPEED)
    runner_state = make_agent_state(runner_pos, chaser_pos, RUNNER_SPEED)

    chaser_action = chaser_ai.choose_action(chaser_state, epsilon)
    runner_action = runner_ai.choose_action(runner_state, epsilon)

    new_chaser_pos, chaser_hit_wall = move_position(
        chaser_pos,
        chaser_action,
        CHASER_SPEED,
    )

    new_runner_pos, runner_hit_wall = move_position(
        runner_pos,
        runner_action,
        RUNNER_SPEED,
    )

    new_distance = distance(new_chaser_pos, new_runner_pos)

    # -----------------------------
    # 보상 설계
    # -----------------------------
    # 술래:
    # - 가까워지면 보상
    # - 멀어지면 벌점
    #
    # 도망자:
    # - 멀어지면 보상
    # - 가까워지면 벌점
    # - 생존할 때마다 작은 보상
    # -----------------------------
    chaser_reward = -0.01 + (old_distance - new_distance) * 0.02
    runner_reward = 0.01 + (new_distance - old_distance) * 0.02

    # 벽에 계속 부딪히는 행동은 강한 벌점으로 처리한다.
    # 그래야 학습이 진행될수록 벽에 박는 행동의 Q값이 확실히 낮아진다.
    if chaser_hit_wall:
        chaser_reward -= WALL_HIT_PENALTY
        if distance(old_chaser_pos, new_chaser_pos) < 1e-6:
            chaser_reward -= STUCK_PENALTY

    if runner_hit_wall:
        runner_reward -= WALL_HIT_PENALTY
        if distance(old_runner_pos, new_runner_pos) < 1e-6:
            runner_reward -= STUCK_PENALTY

    winner = None
    result_text = "running"

    capture_blocked_by_wall = (
        new_distance <= CAPTURE_RADIUS
        and is_blocked_by_any_wall(new_chaser_pos, new_runner_pos)
    )

    # 술래가 도망자를 잡은 경우
    # 단, 둘 사이를 벽이 막고 있으면 잡지 못한다.
    if can_capture(new_chaser_pos, new_runner_pos):
        chaser_reward += 30.0
        runner_reward -= 30.0
        winner = "chaser"
        result_text = "chaser_win"
    elif capture_blocked_by_wall:
        # 거리상으로는 가까워도 벽 때문에 잡지 못한 상황
        chaser_reward -= 0.50
        runner_reward += 0.50
        result_text = "capture_blocked_by_wall"

    # 도망자가 제한 시간까지 버틴 경우
    elif step_index >= max_steps:
        chaser_reward -= 10.0
        runner_reward += 10.0
        winner = "runner"
        result_text = "runner_win"

    next_chaser_state = make_agent_state(new_chaser_pos, new_runner_pos, CHASER_SPEED)
    next_runner_state = make_agent_state(new_runner_pos, new_chaser_pos, RUNNER_SPEED)

    if training:
        chaser_ai.update_q(
            chaser_state,
            chaser_action,
            chaser_reward,
            next_chaser_state,
        )
        runner_ai.update_q(
            runner_state,
            runner_action,
            runner_reward,
            next_runner_state,
        )

    info = {
        "step": step_index,
        "old_chaser_pos": old_chaser_pos,
        "old_runner_pos": old_runner_pos,
        "new_chaser_pos": new_chaser_pos,
        "new_runner_pos": new_runner_pos,
        "chaser_action": chaser_action,
        "runner_action": runner_action,
        "old_distance": old_distance,
        "new_distance": new_distance,
        "chaser_reward": chaser_reward,
        "runner_reward": runner_reward,
        "winner": winner,
        "result": result_text,
        "capture_blocked_by_wall": capture_blocked_by_wall,
    }

    return new_chaser_pos, new_runner_pos, winner, info


def play_one_episode(chaser_ai, runner_ai, epsilon, training=True, record=False):
    """
    게임 1판 실행.

    training=True:
        Q-table 업데이트 O

    training=False:
        Q-table 업데이트 X

    record=True:
        매 step의 위치와 행동을 기록해서 반환
    """
    chaser_pos, runner_pos = get_start_positions()
    trajectory = []
    max_steps = get_max_steps()

    for step in range(1, max_steps + 1):
        chaser_pos, runner_pos, winner, info = simulate_one_step(
            chaser_ai,
            runner_ai,
            chaser_pos,
            runner_pos,
            epsilon,
            step,
            max_steps,
            training=training,
        )

        if record:
            trajectory.append(info)

        if winner is not None:
            if training:
                if winner == "chaser":
                    chaser_ai.score += 1
                    runner_ai.score -= 1
                else:
                    runner_ai.score += 1
                    chaser_ai.score -= 1

            return winner, step, trajectory

    return "runner", max_steps, trajectory


def train_episodes_fast(chaser_ai, runner_ai, episodes, total_episodes_done):
    """
    GUI 없이 빠르게 학습한다.

    기본 학습 명령은 GUI로 보여주며 학습하지만,
    너무 많은 판수를 빠르게 누적하고 싶을 때 fast 명령으로 사용한다.
    """
    chaser_wins = 0
    runner_wins = 0
    total_steps = 0

    progress_interval = max(1, episodes // 10)

    for i in range(1, episodes + 1):
        epsilon = get_epsilon(total_episodes_done + i)
        winner, steps, _ = play_one_episode(
            chaser_ai,
            runner_ai,
            epsilon,
            training=True,
            record=False,
        )

        total_steps += steps

        if winner == "chaser":
            chaser_wins += 1
        else:
            runner_wins += 1

        if episodes >= 1000 and i % progress_interval == 0:
            print(
                f"fast 학습 진행: {i}/{episodes} "
                f"({i / episodes * 100:.0f}%) | "
                f"벽 캐시: {len(WALL_SENSOR_CACHE)}"
            )

    print("\n===== 빠른 추가 학습 결과 =====")
    print(f"이번 실행에서 학습한 게임 수: {episodes}")
    print(f"이번 실행 술래 승: {chaser_wins}")
    print(f"이번 실행 도망자 승: {runner_wins}")
    print(f"이번 실행 평균 게임 길이: {total_steps / episodes:.2f}초")
    print(f"현재 epsilon: {get_epsilon(total_episodes_done + episodes):.3f}")

    return total_episodes_done + episodes


def evaluate(chaser_ai, runner_ai, total_episodes_done, games=200):
    """현재 AI의 성능을 평가한다. 평가 중에는 학습하지 않는다."""
    chaser_wins = 0
    runner_wins = 0
    total_steps = 0

    for _ in range(games):
        winner, steps, _ = play_one_episode(
            chaser_ai,
            runner_ai,
            epsilon=0.0,
            training=False,
            record=False,
        )

        total_steps += steps

        if winner == "chaser":
            chaser_wins += 1
        else:
            runner_wins += 1

    print("\n===== 평가 결과 =====")
    print(f"지금까지 누적 학습한 게임 수: {total_episodes_done}")
    print(f"평가 게임 수: {games}")
    print(f"도망자가 버텨야 하는 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"시작 위치 모드: {START_MODE}")
    print(f"벽 개수: {len(WALLS)}")
    print(f"술래 승률: {chaser_wins / games * 100:.1f}%")
    print(f"도망자 승률: {runner_wins / games * 100:.1f}%")
    print(f"평균 게임 길이: {total_steps * SIM_SECONDS_PER_STEP:.2f}초")


def format_pos(pos):
    return f"({pos[0]:6.1f}, {pos[1]:6.1f})"


def demo_text(chaser_ai, runner_ai):
    """학습된 AI끼리 게임 1판을 텍스트로 자세히 보여준다."""
    winner, steps, trajectory = play_one_episode(
        chaser_ai,
        runner_ai,
        epsilon=0.0,
        training=False,
        record=True,
    )

    print("\n===== 연속 좌표 경기장 텍스트 시연 =====")
    print(f"경기장 크기: {ARENA_SIZE} x {ARENA_SIZE}")
    print(f"술래 속도: {CHASER_SPEED}, 도망자 속도: {RUNNER_SPEED}")
    print(f"잡힘 반경: {CAPTURE_RADIUS}")
    print(f"도망자가 버텨야 하는 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"시작 위치 모드: {START_MODE}")
    print(f"벽 개수: {len(WALLS)}")
    print()

    for item in trajectory:
        elapsed_seconds = item["step"] * SIM_SECONDS_PER_STEP
        remain_seconds = max(0.0, get_survive_seconds() - elapsed_seconds)

        print(f"===== 시간 {elapsed_seconds:.0f}초 / 남은 시간 {remain_seconds:.0f}초 =====")
        print(f"술래   : {format_pos(item['old_chaser_pos'])} -> {format_pos(item['new_chaser_pos'])} / 행동: {ACTION_NAMES[item['chaser_action']]}")
        print(f"도망자 : {format_pos(item['old_runner_pos'])} -> {format_pos(item['new_runner_pos'])} / 행동: {ACTION_NAMES[item['runner_action']]}")
        print(f"거리 변화: {item['old_distance']:.2f} -> {item['new_distance']:.2f}")
        print(f"보상: 술래 {item['chaser_reward']:.3f}, 도망자 {item['runner_reward']:.3f}")

        if item.get("capture_blocked_by_wall", False):
            print("상황 해석: 거리는 충분히 가깝지만 벽이 사이를 막고 있어 잡지 못했다.")
        elif item["new_distance"] <= CAPTURE_RADIUS:
            print("상황 해석: 술래가 도망자를 잡았다.")
        elif item["new_distance"] < item["old_distance"]:
            print("상황 해석: 술래가 도망자에게 가까워졌다.")
        elif item["new_distance"] > item["old_distance"]:
            print("상황 해석: 도망자가 거리를 벌렸다.")
        else:
            print("상황 해석: 거리가 거의 변하지 않았다.")

        print()

    if winner == "chaser":
        print(f"결과: 술래 승리! {steps * SIM_SECONDS_PER_STEP:.0f}초 만에 잡았다.")
    else:
        print(f"결과: 도망자 승리! {SURVIVE_MINUTES:.2f}분 동안 버텼다.")


def create_gui_window(title):
    """tkinter GUI 기본 창과 도형을 만든다."""
    import tkinter as tk

    canvas_size = 640
    padding = 50
    scale = (canvas_size - 2 * padding) / ARENA_SIZE

    root = tk.Tk()
    root.title(title)

    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
    canvas.pack()

    info_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
    info_label.pack()

    left = padding
    top = padding
    right = padding + ARENA_SIZE * scale
    bottom = padding + ARENA_SIZE * scale

    canvas.create_rectangle(left, top, right, bottom, width=3)

    wall_objs = []

    def to_canvas(pos):
        x = padding + pos[0] * scale
        y = padding + (ARENA_SIZE - pos[1]) * scale
        return x, y

    for wall in WALLS:
        x1, y1 = to_canvas((wall["x1"], wall["y1"]))
        x2, y2 = to_canvas((wall["x2"], wall["y2"]))
        wall_width = max(1, int(wall["thickness"] * scale))
        obj = canvas.create_line(x1, y1, x2, y2, width=wall_width, fill="black")
        wall_objs.append(obj)

    chaser_radius = 8
    runner_radius = 8

    chaser_obj = canvas.create_oval(0, 0, 0, 0, fill="red")
    runner_obj = canvas.create_oval(0, 0, 0, 0, fill="blue")
    capture_obj = canvas.create_oval(0, 0, 0, 0, outline="red", dash=(4, 4))

    def update_circle(obj, pos, radius):
        x, y = to_canvas(pos)
        canvas.coords(obj, x - radius, y - radius, x + radius, y + radius)

    def update_capture_circle(pos):
        x, y = to_canvas(pos)
        r = CAPTURE_RADIUS * scale
        canvas.coords(capture_obj, x - r, y - r, x + r, y + r)

    gui = {
        "tk": tk,
        "root": root,
        "canvas": canvas,
        "info_label": info_label,
        "chaser_obj": chaser_obj,
        "runner_obj": runner_obj,
        "capture_obj": capture_obj,
        "wall_objs": wall_objs,
        "chaser_radius": chaser_radius,
        "runner_radius": runner_radius,
        "update_circle": update_circle,
        "update_capture_circle": update_capture_circle,
    }

    return gui


def demo_gui(chaser_ai, runner_ai):
    """
    학습 없이 현재 AI의 움직임을 GUI로 1판 보여준다.
    """
    try:
        create_gui_window
        import tkinter  # noqa: F401
    except Exception:
        print("tkinter를 사용할 수 없어 GUI 시연을 실행할 수 없다.")
        return

    gui = create_gui_window("Continuous Tag AI Simulation - Demo")

    winner, steps, trajectory = play_one_episode(
        chaser_ai,
        runner_ai,
        epsilon=0.0,
        training=False,
        record=True,
    )

    root = gui["root"]
    info_label = gui["info_label"]

    def animate(index=0):
        if index >= len(trajectory):
            if winner == "chaser":
                info_label.config(text=f"결과: 술래 승리! {steps * SIM_SECONDS_PER_STEP:.0f}초 만에 잡았다.")
            else:
                info_label.config(text=f"결과: 도망자 승리! {SURVIVE_MINUTES:.2f}분 동안 버텼다.")
            return

        item = trajectory[index]
        chaser_pos = item["new_chaser_pos"]
        runner_pos = item["new_runner_pos"]

        gui["update_circle"](gui["chaser_obj"], chaser_pos, gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], runner_pos, gui["runner_radius"])
        gui["update_capture_circle"](chaser_pos)

        elapsed_seconds = item["step"] * SIM_SECONDS_PER_STEP
        remain_seconds = max(0.0, get_survive_seconds() - elapsed_seconds)

        info_label.config(
            text=(
                f"Demo | 시간: {elapsed_seconds:.0f}초 / 남은 시간: {remain_seconds:.0f}초\n"
                f"거리: {item['new_distance']:.2f} | "
                f"술래: {ACTION_NAMES[item['chaser_action']]} | "
                f"도망자: {ACTION_NAMES[item['runner_action']]}\n"
                f"배속: {TRAIN_SPEED_MULTIPLIER:.1f}x | "
                f"생존 목표: {SURVIVE_MINUTES:.2f}분 | 시작 모드: {START_MODE} | 벽: {len(WALLS)}개"
            )
        )

        root.after(get_animation_delay_ms(), lambda: animate(index + 1))

    if trajectory:
        gui["update_circle"](gui["chaser_obj"], trajectory[0]["old_chaser_pos"], gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], trajectory[0]["old_runner_pos"], gui["runner_radius"])
        gui["update_capture_circle"](trajectory[0]["old_chaser_pos"])

    animate()
    root.mainloop()


def train_episodes_visual(chaser_ai, runner_ai, episodes, total_episodes_done):
    """
    사용자가 입력한 학습 판수만큼 GUI로 보여주면서 실제 학습한다.

    이 함수에서는 매 step마다:
    1. 술래와 도망자가 움직임
    2. Q-table이 즉시 업데이트됨
    3. 화면이 갱신됨

    즉, 단순 시연이 아니라 실제 학습 과정이다.
    """
    try:
        import tkinter  # noqa: F401
    except Exception:
        print("tkinter를 사용할 수 없어 GUI 학습을 실행할 수 없다.")
        print("대신 빠른 학습으로 진행한다.")
        return train_episodes_fast(chaser_ai, runner_ai, episodes, total_episodes_done)

    gui = create_gui_window("Continuous Tag AI Simulation - Visual Training")

    root = gui["root"]
    info_label = gui["info_label"]

    state = {
        "completed": 0,
        "total_episodes_done": total_episodes_done,
        "chaser_wins": 0,
        "runner_wins": 0,
        "total_steps": 0,
        "current_step": 0,
        "chaser_pos": None,
        "runner_pos": None,
        "epsilon": None,
        "closed": False,
    }

    def save_and_close():
        state["closed"] = True
        save_learning_data(chaser_ai, runner_ai, state["total_episodes_done"])
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", save_and_close)

    def start_episode():
        if state["completed"] >= episodes:
            save_learning_data(chaser_ai, runner_ai, state["total_episodes_done"])

            avg_seconds = 0.0
            if state["completed"] > 0:
                avg_seconds = state["total_steps"] * SIM_SECONDS_PER_STEP / state["completed"]

            info_label.config(
                text=(
                    f"시각화 학습 완료\n"
                    f"이번 실행 학습 게임 수: {state['completed']}\n"
                    f"술래 승: {state['chaser_wins']} | 도망자 승: {state['runner_wins']}\n"
                    f"평균 게임 길이: {avg_seconds:.2f}초\n"
                    f"누적 학습 게임 수: {state['total_episodes_done']}"
                )
            )
            return

        state["chaser_pos"], state["runner_pos"] = get_start_positions()
        state["current_step"] = 0

        # 이번 에피소드에서 사용할 epsilon
        state["epsilon"] = get_epsilon(state["total_episodes_done"] + 1)

        gui["update_circle"](gui["chaser_obj"], state["chaser_pos"], gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], state["runner_pos"], gui["runner_radius"])
        gui["update_capture_circle"](state["chaser_pos"])

        root.after(300, animate_step)

    def animate_step():
        max_steps = get_max_steps()
        state["current_step"] += 1

        chaser_pos, runner_pos, winner, info = simulate_one_step(
            chaser_ai,
            runner_ai,
            state["chaser_pos"],
            state["runner_pos"],
            state["epsilon"],
            state["current_step"],
            max_steps,
            training=True,
        )

        state["chaser_pos"] = chaser_pos
        state["runner_pos"] = runner_pos

        gui["update_circle"](gui["chaser_obj"], chaser_pos, gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], runner_pos, gui["runner_radius"])
        gui["update_capture_circle"](chaser_pos)

        elapsed_seconds = state["current_step"] * SIM_SECONDS_PER_STEP
        remain_seconds = max(0.0, get_survive_seconds() - elapsed_seconds)

        info_label.config(
            text=(
                f"시각화 학습 중\n"
                f"현재 판: {state['completed'] + 1} / {episodes} | "
                f"누적 학습 게임 수: {state['total_episodes_done']}\n"
                f"시간: {elapsed_seconds:.0f}초 / 남은 시간: {remain_seconds:.0f}초 | "
                f"생존 목표: {SURVIVE_MINUTES:.2f}분\n"
                f"거리: {info['new_distance']:.2f} | "
                f"술래 행동: {ACTION_NAMES[info['chaser_action']]} | "
                f"도망자 행동: {ACTION_NAMES[info['runner_action']]}\n"
                f"epsilon: {state['epsilon']:.3f} | "
                f"배속: {TRAIN_SPEED_MULTIPLIER:.1f}x | "
                f"벽: {len(WALLS)}개 | "
                f"술래 승: {state['chaser_wins']} | 도망자 승: {state['runner_wins']}"
            )
        )

        if winner is not None:
            state["completed"] += 1
            state["total_episodes_done"] += 1
            state["total_steps"] += state["current_step"]

            if winner == "chaser":
                chaser_ai.score += 1
                runner_ai.score -= 1
                state["chaser_wins"] += 1
            else:
                runner_ai.score += 1
                chaser_ai.score -= 1
                state["runner_wins"] += 1

            # 한 판이 끝났을 때 잠깐 결과를 보여준 후 다음 판 시작
            result = "술래 승리" if winner == "chaser" else "도망자 승리"
            info_label.config(
                text=(
                    f"{result}\n"
                    f"완료된 판: {state['completed']} / {episodes}\n"
                    f"이번 판 진행 시간: {state['current_step'] * SIM_SECONDS_PER_STEP:.0f}초\n"
                    f"다음 판을 시작한다..."
                )
            )

            root.after(max(100, int(700 / max(1.0, TRAIN_SPEED_MULTIPLIER))), start_episode)
            return

        root.after(get_animation_delay_ms(), animate_step)

    start_episode()
    root.mainloop()

    return state["total_episodes_done"]


def save_learning_data(chaser_ai, runner_ai, total_episodes_done):
    """현재 학습 데이터와 설정값을 파일로 저장한다."""
    data = {
        "total_episodes_done": total_episodes_done,
        "chaser_score": chaser_ai.score,
        "runner_score": runner_ai.score,
        "chaser_q_table": dict(chaser_ai.q_table),
        "runner_q_table": dict(runner_ai.q_table),
        "settings": {
            "arena_size": ARENA_SIZE,
            "capture_radius": CAPTURE_RADIUS,
            "sim_seconds_per_step": SIM_SECONDS_PER_STEP,
            "survive_minutes": SURVIVE_MINUTES,
            "chaser_speed": CHASER_SPEED,
            "runner_speed": RUNNER_SPEED,
            "train_speed_multiplier": TRAIN_SPEED_MULTIPLIER,
            "start_mode": START_MODE,
            "walls": WALLS,
            "wall_sensor_lookahead": WALL_SENSOR_LOOKAHEAD,
            "wall_hit_penalty": WALL_HIT_PENALTY,
            "stuck_penalty": STUCK_PENALTY,
            "wall_sensor_cache_grid": WALL_SENSOR_CACHE_GRID,
        },
    }

    with open(SAVE_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\n학습 데이터 저장 완료: {SAVE_FILE}")


def load_learning_data(chaser_ai, runner_ai):
    """저장된 학습 데이터가 있으면 불러온다."""
    global SURVIVE_MINUTES
    global TRAIN_SPEED_MULTIPLIER
    global START_MODE
    global WALLS

    if not os.path.exists(SAVE_FILE):
        print("저장된 학습 데이터가 없다. 새 학습을 시작한다.")
        return 0

    with open(SAVE_FILE, "rb") as f:
        data = pickle.load(f)

    chaser_ai.score = data.get("chaser_score", 0)
    runner_ai.score = data.get("runner_score", 0)

    chaser_ai.q_table.clear()
    runner_ai.q_table.clear()

    chaser_ai.q_table.update(data.get("chaser_q_table", {}))
    runner_ai.q_table.update(data.get("runner_q_table", {}))

    settings = data.get("settings", {})
    SURVIVE_MINUTES = float(settings.get("survive_minutes", SURVIVE_MINUTES))
    TRAIN_SPEED_MULTIPLIER = float(settings.get("train_speed_multiplier", TRAIN_SPEED_MULTIPLIER))
    START_MODE = settings.get("start_mode", START_MODE)

    if START_MODE not in ("random", "fixed"):
        START_MODE = "random"

    WALLS.clear()
    loaded_walls = settings.get("walls", [])
    for wall in loaded_walls:
        try:
            WALLS.append(
                clamp_wall_to_arena(
                    make_wall(
                        wall["x1"],
                        wall["y1"],
                        wall["x2"],
                        wall["y2"],
                        wall.get("thickness", DEFAULT_WALL_THICKNESS),
                    )
                )
            )
        except Exception:
            pass

    total_episodes_done = data.get("total_episodes_done", 0)

    print("저장된 학습 데이터를 불러왔다.")
    print(f"누적 학습 게임 수: {total_episodes_done}")
    print(f"누적 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")
    print(f"도망자 생존 목표 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"현재 시각화 배속: {TRAIN_SPEED_MULTIPLIER:.1f}x")
    clear_wall_sensor_cache()

    print(f"현재 시작 위치 모드: {START_MODE}")
    print(f"현재 벽 개수: {len(WALLS)}")
    if len(WALLS) > 0:
        print("저장 파일에서 벽 정보를 불러왔기 때문에 첫 화면에 벽 개수가 표시된다.")

    return total_episodes_done


def reset_learning_data(chaser_ai, runner_ai):
    """저장 파일, 현재 메모리의 학습 데이터, 벽 정보를 모두 초기화한다."""
    chaser_ai.clear_learning_data()
    runner_ai.clear_learning_data()
    WALLS.clear()

    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)

    print("\n전체 학습 데이터를 초기화했다.")
    print("Q-table, 점수, 저장 파일, 벽 정보를 모두 삭제했다.")
    print("다음 학습은 완전히 새 상태에서 시작된다.")

    return 0


def command_to_episodes(command):
    """일반 학습 명령어를 학습 판수로 변환한다."""
    if command == "":
        return 1

    preset_commands = {
        "1": 1,
        "train1": 1,
        "10": 10,
        "train10": 10,
        "100": 100,
        "train100": 100,
        "1000": 1000,
        "train1000": 1000,
    }

    if command in preset_commands:
        return preset_commands[command]

    if command.isdigit():
        value = int(command)
        if value > 0:
            return value

    return None


def command_to_fast_episodes(command):
    """
    빠른 학습 명령어를 판수로 변환한다.

    예:
    fast 1000
    f 1000
    """
    parts = command.split()

    if len(parts) != 2:
        return None

    if parts[0] not in ("fast", "f"):
        return None

    if not parts[1].isdigit():
        return None

    value = int(parts[1])
    if value <= 0:
        return None

    return value


def handle_speed_command(command):
    """
    배속 변경 명령 처리.

    예:
    speed 20
    speed 100
    """
    global TRAIN_SPEED_MULTIPLIER

    parts = command.split()

    if len(parts) != 2:
        return False

    if parts[0] not in ("speed", "배속"):
        return False

    try:
        value = float(parts[1])
    except ValueError:
        print("배속은 숫자로 입력해야 한다. 예: speed 20")
        return True

    if value <= 0:
        print("배속은 0보다 커야 한다.")
        return True

    TRAIN_SPEED_MULTIPLIER = value
    print(f"시각화 배속을 {TRAIN_SPEED_MULTIPLIER:.1f}x로 변경했다.")
    return True


def handle_minutes_command(command):
    """
    도망자 생존 목표 시간 변경 명령 처리.

    예:
    minutes 1
    minutes 0.5
    time 2
    """
    global SURVIVE_MINUTES

    parts = command.split()

    if len(parts) != 2:
        return False

    if parts[0] not in ("minutes", "minute", "time", "m", "분"):
        return False

    try:
        value = float(parts[1])
    except ValueError:
        print("시간은 숫자로 입력해야 한다. 예: minutes 1.5")
        return True

    if value <= 0:
        print("시간은 0보다 커야 한다.")
        return True

    SURVIVE_MINUTES = value
    print(f"도망자 생존 목표 시간을 {SURVIVE_MINUTES:.2f}분으로 변경했다.")
    print(f"현재 설정에서는 총 {get_max_steps()}초 동안 버티면 도망자 승리다.")
    return True


def handle_start_command(command):
    """
    시작 위치 모드 변경 명령 처리.

    예:
    start random
    start fixed
    시작 random
    시작 fixed
    """
    global START_MODE
    global WALLS

    parts = command.split()

    if len(parts) != 2:
        return False

    if parts[0] not in ("start", "시작"):
        return False

    mode = parts[1]

    if mode not in ("random", "fixed", "랜덤", "고정"):
        print("시작 위치 모드는 random 또는 fixed만 가능하다.")
        print("예: start random")
        print("예: start fixed")
        return True

    if mode == "랜덤":
        mode = "random"
    elif mode == "고정":
        mode = "fixed"

    START_MODE = mode

    if START_MODE == "random":
        print("시작 위치 모드를 random으로 변경했다.")
        print("이제 매 판마다 술래와 도망자의 시작 위치가 랜덤으로 정해진다.")
    else:
        print("시작 위치 모드를 fixed로 변경했다.")
        print(f"술래 고정 시작 위치: {FIXED_CHASER_POS}")
        print(f"도망자 고정 시작 위치: {FIXED_RUNNER_POS}")

    return True


def handle_wall_command(command):
    """
    벽 관련 명령 처리.

    예:
    wall add 250 100 250 400
    wall add 250 100 250 400 20
    wall list
    wall clear
    wall remove 1
    """
    parts = command.split()

    if not parts:
        return False

    if parts[0] not in ("wall", "벽"):
        return False

    if len(parts) == 1:
        print("벽 명령어 사용법:")
        print("wall add x1 y1 x2 y2 [thickness]")
        print("wall list")
        print("wall clear")
        print("wall remove 번호")
        return True

    action = parts[1]

    if action in ("add", "추가"):
        if len(parts) not in (6, 7):
            print("벽 추가 형식이 올바르지 않다.")
            print("예: wall add 250 100 250 400")
            print("예: wall add 250 100 250 400 20")
            return True

        try:
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            thickness = float(parts[6]) if len(parts) == 7 else DEFAULT_WALL_THICKNESS
        except ValueError:
            print("좌표와 두께는 숫자로 입력해야 한다.")
            return True

        add_wall(x1, y1, x2, y2, thickness)
        return True

    if action in ("list", "목록"):
        list_walls()
        return True

    if action in ("clear", "전체삭제", "초기화"):
        clear_walls()
        return True

    if action in ("remove", "delete", "삭제"):
        if len(parts) != 3 or not parts[2].isdigit():
            print("삭제할 벽 번호를 입력해야 한다. 예: wall remove 1")
            return True

        remove_wall(int(parts[2]))
        return True

    print("알 수 없는 wall 명령어다.")
    print("wall add / wall list / wall clear / wall remove 를 사용할 수 있다.")
    return True


def handle_wall_cache_command(command):
    """
    벽 감지 캐시 정밀도 설정.

    예:
    wallcache 10  -> 더 정밀하지만 느림
    wallcache 25  -> 더 빠르지만 대략적
    """
    global WALL_SENSOR_CACHE_GRID

    parts = command.split()

    if len(parts) != 2:
        return False

    if parts[0] not in ("wallcache", "cachegrid", "벽캐시"):
        return False

    try:
        value = float(parts[1])
    except ValueError:
        print("벽 캐시 격자 크기는 숫자로 입력해야 한다. 예: wallcache 20")
        return True

    if value < 3:
        print("벽 캐시 격자 크기가 너무 작다. 3 이상을 권장한다.")
        return True

    WALL_SENSOR_CACHE_GRID = value
    clear_wall_sensor_cache()

    print(f"벽 감지 캐시 격자 크기를 {WALL_SENSOR_CACHE_GRID:.1f}로 변경했다.")
    print("값이 클수록 fast 학습은 빨라지지만 벽 감지는 더 대략적이다.")
    return True


def print_menu():
    print("\n===== 명령어 =====")
    print("Enter 또는 1        : 1판 시각화 학습 + 저장")
    print("10 또는 train10     : 10판 시각화 학습 + 저장")
    print("100 또는 train100   : 100판 시각화 학습 + 저장")
    print("1000 또는 train1000 : 1000판 시각화 학습 + 저장")
    print("숫자 입력           : 해당 숫자만큼 시각화 학습 + 저장")
    print("fast 숫자           : GUI 없이 빠르게 추가 학습. 예: fast 1000")
    print("demo                : 학습 없이 현재 AI로 텍스트 시연")
    print("visual              : 학습 없이 현재 AI로 GUI 시연")
    print("eval                : 학습 없이 현재 AI 승률 평가")
    print("speed 숫자          : 학습/시연 화면 배속 변경. 예: speed 50")
    print("minutes 숫자        : 도망자 생존 목표 시간 변경. 예: minutes 0.5")
    print("start random        : 매 판 랜덤 위치에서 시작")
    print("start fixed         : 매 판 고정 위치에서 시작")
    print("wall add x1 y1 x2 y2 [두께] : 벽 추가. 예: wall add 250 100 250 400 20")
    print("wall list           : 현재 벽 목록 보기")
    print("wall remove 번호    : 특정 벽 삭제")
    print("wall clear          : 모든 벽 삭제")
    print("wallcache 숫자      : 벽 감지 캐시 크기 설정. 예: wallcache 25")
    print("reset               : 저장된 학습 데이터 전체 초기화")
    print("quit                : 저장 후 종료")
    print("\n----- 현재 설정 -----")
    print(f"도망자 생존 목표 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"시각화 배속: {TRAIN_SPEED_MULTIPLIER:.1f}x")
    print(f"시작 위치 모드: {START_MODE}")
    print(f"벽 개수: {len(WALLS)}")
    print(f"벽 감지 거리 배율: {WALL_SENSOR_LOOKAHEAD}x")
    print(f"벽 감지 캐시 격자: {WALL_SENSOR_CACHE_GRID}")
    print(f"벽 감지 캐시 저장 수: {len(WALL_SENSOR_CACHE)}")
    print(f"벽 충돌 벌점: -{WALL_HIT_PENALTY}, 멈춤 추가 벌점: -{STUCK_PENALTY}")
    print(f"잡힘 반경: {CAPTURE_RADIUS}")
    print(f"술래 속도: {CHASER_SPEED}, 도망자 속도: {RUNNER_SPEED}")


def main():
    random.seed()

    chaser_ai = QLearningAgent("술래 AI")
    runner_ai = QLearningAgent("도망자 AI")

    total_episodes_done = load_learning_data(chaser_ai, runner_ai)

    while True:
        print_menu()
        command = input("\n명령어 입력: ").strip().lower()

        if handle_speed_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        if handle_minutes_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        if handle_start_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        if handle_wall_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        if handle_wall_cache_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        fast_episodes = command_to_fast_episodes(command)
        if fast_episodes is not None:
            total_episodes_done = train_episodes_fast(
                chaser_ai,
                runner_ai,
                fast_episodes,
                total_episodes_done,
            )
            print(f"\n누적 학습 게임 수: {total_episodes_done}")
            print(f"누적 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        episodes = command_to_episodes(command)
        if episodes is not None:
            print(f"\n{episodes}판을 시각화하면서 학습한다.")
            print("GUI 창이 열리면 술래와 도망자의 실제 학습 움직임이 보인다.")
            print("창을 닫으면 현재까지 진행된 학습 데이터가 저장된다.")

            total_episodes_done = train_episodes_visual(
                chaser_ai,
                runner_ai,
                episodes,
                total_episodes_done,
            )

            print(f"\n누적 학습 게임 수: {total_episodes_done}")
            print(f"누적 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        if command == "demo":
            demo_text(chaser_ai, runner_ai)

        elif command == "visual":
            demo_gui(chaser_ai, runner_ai)

        elif command == "eval":
            evaluate(chaser_ai, runner_ai, total_episodes_done, games=200)

        elif command == "reset":
            total_episodes_done = reset_learning_data(chaser_ai, runner_ai)

        elif command == "quit":
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            print("프로그램을 종료한다.")
            break

        else:
            print("알 수 없는 명령어다.")


if __name__ == "__main__":
    main()
