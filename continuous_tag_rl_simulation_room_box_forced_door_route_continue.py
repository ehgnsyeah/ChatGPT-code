import os
import pickle
import random
import math
import heapq
from collections import defaultdict

# =========================================================
# 정사각형 경기장 술래잡기 강화학습 시뮬레이션 - Room & Box 버전
# - 연속 좌표 기반 자유 이동
# - Q-learning 기반 술래 AI / 도망자 AI
# - 학습 데이터 저장/불러오기
# - 학습 과정 GUI 시각화
# - 도망자 생존 시간을 '분' 단위로 설정
# - 시작 위치 random/fixed 선택 가능
# - 사용자 벽 + 기본 방 구조 지원
# - 밀어서 옮길 수 있는 상자 1개 지원
# - 도망자가 상자를 이용해 술래의 추격 경로를 방해하는 전략을 학습하도록 보상 설계 강화
# - 술래가 제자리걸음/헛도는 상황에서만 A*를 사용하는 길찾기 보조 추가
# - AI에게 벽/상자/외곽 정보를 미리 알려 주는 map-aware action masking 추가
# - AI 몸 크기만큼 벽/외곽과 간격 유지
# - 술래가 방 벽 앞에서 헛돌지 않도록 출입구 waypoint 사용
# - 방 안 도망자를 쫓는 술래는 Q-table보다 출입구 우회 경로를 우선 적용
# =========================================================

# -----------------------------
# 경기장 설정
# -----------------------------
ARENA_SIZE = 500.0
CAPTURE_RADIUS = 18.0
SIM_SECONDS_PER_STEP = 1.0
SURVIVE_MINUTES = 1.0

# 매 판 시작 후 이 시간 동안 술래는 움직이지 못한다.
# 단위: 초
CHASER_FREEZE_SECONDS = 10.0

CHASER_SPEED = 20.0
RUNNER_SPEED = 17.0
MIN_START_DISTANCE = 150.0

START_MODE = "fixed"
# 기본 고정 시작 위치:
# 술래는 방 밖 아래쪽, 도망자는 방 안쪽에서 시작한다.
# 이렇게 해야 도망자가 상자를 이용해 문을 막는 전략을 더 잘 학습할 수 있다.
FIXED_CHASER_POS = (250.0, 80.0)
FIXED_RUNNER_POS = (250.0, 455.0)

# -----------------------------
# 방 구조 설정
# -----------------------------
# 전체 경기장 면적의 1/4 -> 한 변 250인 정사각형 방
ROOM_ENABLED = True
ROOM_SIZE = ARENA_SIZE / 2.0  # 250
ROOM_LEFT = (ARENA_SIZE - ROOM_SIZE) / 2.0  # 125
ROOM_RIGHT = ROOM_LEFT + ROOM_SIZE          # 375
ROOM_TOP = ARENA_SIZE                       # 500, 경기장 위쪽 외벽과 맞닿음
ROOM_BOTTOM = ROOM_TOP - ROOM_SIZE          # 250

# 좌/우 벽 각각 출입구 1개
DOOR_SIZE = 80.0
DOOR_CENTER_Y = ROOM_BOTTOM + ROOM_SIZE * 0.55
DOOR_BOTTOM = DOOR_CENTER_Y - DOOR_SIZE / 2.0
DOOR_TOP = DOOR_CENTER_Y + DOOR_SIZE / 2.0
DEFAULT_WALL_THICKNESS = 14.0

# -----------------------------
# 상자 설정
# -----------------------------
BOX_ENABLED = True
BOX_SIZE = 72.0  # 출입구 하나를 거의 막을 수 있을 정도
# 매 판이 시작될 때 상자가 놓이는 기본 위치.
# 실행 중에는 boxstart x y 명령어로 바꿀 수 있다.
BOX_START_POS = (
    (ROOM_LEFT + ROOM_RIGHT) / 2.0,
    ROOM_BOTTOM + ROOM_SIZE * 0.56,
)
BOX_PUSH_MARGIN = 2.0

# AI를 점이 아니라 작은 원처럼 취급하기 위한 충돌 반경.
# 상자가 이 반경 안으로 겹치면 AI가 상자 안에 갇히는 것으로 보고 막는다.
AGENT_COLLISION_RADIUS = 8.0

BOX_PUSH_REWARD = 0.2
BOX_STUCK_PENALTY = 2.0

# 도망자가 상자를 이용해 술래를 방해하도록 유도하는 보상
# 핵심 목표:
# - 상자를 문으로 가져가는 것이 아니라,
# - 상자가 술래와 도망자 사이에 놓이도록 학습시키는 것이다.

# 상자가 술래-도망자 사이의 직선 경로에 가까워질 때 주는 보상
BOX_INTERFERENCE_PROGRESS_REWARD = 5.0

# 상자가 실제로 술래와 도망자 사이의 시야/경로를 막았을 때 주는 큰 보상
BOX_INTERFERENCE_REWARD = 10.0

# 상자가 계속 술래와 도망자 사이를 막고 있을 때 매 step 주는 유지 보상
BOX_INTERFERENCE_STEP_REWARD = 0.35

# 술래가 가까운 상황에서 상자로 가로막는 데 성공했을 때 추가 보상
BOX_CLOSE_THREAT_BONUS = 4.0

# 술래가 상자 때문에 잡지 못했을 때 도망자에게 주는 보상
CAPTURE_BLOCKED_BY_BOX_REWARD = 8.0

# 술래가 상자를 밀 때는 작은 벌점
# 술래가 상자를 치우는 행동을 너무 선호하지 않도록 하기 위한 값이다.
CHASER_BOX_PUSH_PENALTY = 0.10

# -----------------------------
# 학습 / 보상 설정
# -----------------------------
TRAIN_SPEED_MULTIPLIER = 20.0
ALPHA = 0.20
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 20000
ANGLE_BINS = 8
DISTANCE_BINS = 8
WALL_MARGIN = 80.0
WALL_SENSOR_LOOKAHEAD = 2.0
WALL_HIT_PENALTY = 3.0
STUCK_PENALTY = 2.0
WALL_SENSOR_CACHE_GRID = 15.0
WALL_SENSOR_CACHE = {}

# -----------------------------
# 술래 길찾기 강화 설정
# -----------------------------
# True이면 술래가 Q-table만 쓰는 것이 아니라,
# A* 기반 길찾기 보조 알고리즘을 함께 사용한다.
CHASER_PATHFINDING_ENABLED = True

# 0.0~1.0 사이 값.
# 1.0에 가까울수록 술래가 길찾기 알고리즘을 더 자주 따른다.
CHASER_PATHFINDING_RATE = 0.85

# A* 길찾기에서 사용하는 격자 간격.
# 실제 게임은 여전히 연속 좌표이고, 길찾기 계산만 이 간격으로 대략화한다.
PATH_GRID_SIZE = 25.0

# 길찾기 결과 캐시
PATHFINDING_CACHE = {}

# 술래 길찾기 모드
# "fast": A*를 거의 쓰지 않고 8방향 빠른 장애물 회피 추격 사용
# "astar": 기존처럼 A*를 적극적으로 사용
CHASER_PATH_MODE = "fast"

# fast 모드에서 A* fallback을 사용할지 여부
CHASER_ASTAR_FALLBACK_ENABLED = True

# fast 모드에서 A* fallback을 아주 가끔만 사용하기 위한 확률
CHASER_ASTAR_FALLBACK_RATE = 0.03

# -----------------------------
# 술래 제자리걸음 / 헛돎 감지 설정
# -----------------------------
# True이면 술래가 최근 몇 step 동안 도망자와의 거리를 충분히 줄이지 못하거나,
# 비슷한 위치를 왔다갔다 하는 경우 A*를 강제로 사용한다.
CHASER_STUCK_DETECTION_ENABLED = True

# 최근 몇 step을 보고 술래가 헛도는지 판단할지
CHASER_STUCK_WINDOW_STEPS = 8

# 최근 window 동안 이 값보다 거리를 덜 줄이면 헛도는 상황으로 본다.
CHASER_STUCK_MIN_PROGRESS = 8.0

# 술래 위치를 이 크기의 구역으로 묶어서 반복 이동 여부를 판단한다.
CHASER_STUCK_POSITION_GRID = 30.0

# 이 거리보다 가까우면 굳이 A*를 쓰지 않고 직접 추격한다.
CHASER_STUCK_IGNORE_DISTANCE = CAPTURE_RADIUS * 2.5

# 매 episode마다 초기화되는 술래 이동 기록
CHASER_STUCK_HISTORY = []

# -----------------------------
# 지도 사전 인식 설정
# -----------------------------
# True이면 AI가 행동을 고르기 전에 벽/상자/경기장 외곽 때문에
# 갈 수 없는 방향을 미리 알고, 해당 방향을 선택지에서 제외한다.
MAP_AWARE_ACTION_MASKING = True

# True이면 랜덤 탐험 중에도 막힌 방향은 선택하지 않는다.
MAP_AWARE_EXPLORATION = True

# 술래가 상자를 밀 수 있게 둘지 여부.
# True이면 술래도 상자를 밀 수 있지만, 상자 밀기 벌점은 유지된다.
CHASER_CAN_PUSH_BOX = True

# 도망자가 상자를 밀 수 있게 둘지 여부.
RUNNER_CAN_PUSH_BOX = True

# AI가 벽선이나 경기장 외곽선에 올라타지 않도록 하는 여유 거리
# 실제 화면의 빨간/파란 원 반경과 비슷하게 둔다.
WALL_CLEARANCE = AGENT_COLLISION_RADIUS

# 술래가 방 안 도망자를 쫓을 때, 방 벽에 정면으로 박지 않고
# 좌/우 출입구 쪽으로 돌아가게 만드는 빠른 waypoint 보조 기능
CHASER_DOOR_WAYPOINT_ENABLED = True
DOOR_WAYPOINT_OFFSET = 35.0
DOOR_WAYPOINT_REACH_RADIUS = 18.0

# True이면 도망자가 방 안에 있고 술래가 방 밖에 있을 때,
# Q-table보다 먼저 출입구 우회 경로를 강제로 따른다.
# 즉, 술래가 방 아래 벽에 계속 들이받는 상황을 더 강하게 막는다.
CHASER_FORCE_DOOR_ROUTE = True

# 방 아래 벽으로부터 이 거리 안에 있으면 벽 앞에서 맴도는 것으로 보고,
# 좌/우 문 쪽으로 강하게 빠져나가게 한다.
ROOM_WALL_AVOID_BAND = 70.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
SAVE_FILE = os.path.join(BASE_DIR, "continuous_tag_ai_learning_data_room_box_block_chaser.pkl")

# -----------------------------
# 행동 설정: 8방향
# -----------------------------
RAW_ACTIONS = [
    (0.0, 1.0),
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
]
ACTION_NAMES = [
    "UP", "DOWN", "LEFT", "RIGHT",
    "UP_RIGHT", "UP_LEFT", "DOWN_RIGHT", "DOWN_LEFT"
]
ACTIONS = []
for dx, dy in RAW_ACTIONS:
    length = math.hypot(dx, dy)
    ACTIONS.append((dx / length, dy / length))

# -----------------------------
# 전역 장애물 / 상자 상태
# -----------------------------
WALLS = []
BOX_POS = BOX_START_POS


class QLearningAgent:
    def __init__(self, name):
        self.name = name
        self.q_table = defaultdict(lambda: [0.0 for _ in ACTIONS])
        self.score = 0

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))
        q_values = self.q_table[state]
        max_q = max(q_values)
        best = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best)

    def update_q(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        target = reward + GAMMA * next_max_q
        self.q_table[state][action] += ALPHA * (target - current_q)

    def clear_learning_data(self):
        self.q_table.clear()
        self.score = 0


# -----------------------------
# 공통 유틸
# -----------------------------
def get_max_steps():
    survive_seconds = SURVIVE_MINUTES * 60.0
    return max(1, int(math.ceil(survive_seconds / SIM_SECONDS_PER_STEP)))


def get_survive_seconds():
    return SURVIVE_MINUTES * 60.0


def is_chaser_frozen(step_index):
    """
    현재 step에서 술래가 아직 출발 대기 상태인지 확인한다.
    step_index는 1부터 시작하므로 step 1의 시작 시각은 0초다.
    """
    elapsed_before_step = (step_index - 1) * SIM_SECONDS_PER_STEP
    return elapsed_before_step < CHASER_FREEZE_SECONDS


def action_label(action_index):
    """
    action_index를 화면 출력용 문자열로 변환한다.
    술래가 정지 상태일 때는 action_index가 None이다.
    """
    if action_index is None:
        return "FROZEN"
    return ACTION_NAMES[action_index]


def get_animation_delay_ms():
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
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(p1, p2, q1, q2):
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
    if segments_intersect(p1, p2, q1, q2):
        return 0.0
    return min(
        distance_point_to_segment(p1, q1, q2),
        distance_point_to_segment(p2, q1, q2),
        distance_point_to_segment(q1, p1, p2),
        distance_point_to_segment(q2, p1, p2),
    )


# -----------------------------
# 벽 / 방
# -----------------------------
def make_wall(x1, y1, x2, y2, thickness=None, kind="custom"):
    if thickness is None:
        thickness = DEFAULT_WALL_THICKNESS
    return {
        "x1": float(x1), "y1": float(y1),
        "x2": float(x2), "y2": float(y2),
        "thickness": float(thickness),
        "kind": kind,
    }


def wall_points(wall):
    return (wall["x1"], wall["y1"]), (wall["x2"], wall["y2"])


def clamp_wall_to_arena(wall):
    wall["x1"] = clamp(wall["x1"], 0.0, ARENA_SIZE)
    wall["y1"] = clamp(wall["y1"], 0.0, ARENA_SIZE)
    wall["x2"] = clamp(wall["x2"], 0.0, ARENA_SIZE)
    wall["y2"] = clamp(wall["y2"], 0.0, ARENA_SIZE)
    wall["thickness"] = max(1.0, wall["thickness"])
    return wall


def clear_wall_sensor_cache():
    WALL_SENSOR_CACHE.clear()
    if "PATHFINDING_CACHE" in globals():
        PATHFINDING_CACHE.clear()


def rebuild_room_walls():
    global WALLS
    custom_walls = [wall for wall in WALLS if wall.get("kind") != "room"]
    room_walls = []
    if ROOM_ENABLED:
        # 바닥 벽
        room_walls.append(make_wall(ROOM_LEFT, ROOM_BOTTOM, ROOM_RIGHT, ROOM_BOTTOM, kind="room"))
        # 왼쪽 벽 (문 구간 제외)
        room_walls.append(make_wall(ROOM_LEFT, ROOM_BOTTOM, ROOM_LEFT, DOOR_BOTTOM, kind="room"))
        room_walls.append(make_wall(ROOM_LEFT, DOOR_TOP, ROOM_LEFT, ROOM_TOP, kind="room"))
        # 오른쪽 벽 (문 구간 제외)
        room_walls.append(make_wall(ROOM_RIGHT, ROOM_BOTTOM, ROOM_RIGHT, DOOR_BOTTOM, kind="room"))
        room_walls.append(make_wall(ROOM_RIGHT, DOOR_TOP, ROOM_RIGHT, ROOM_TOP, kind="room"))
    WALLS = custom_walls + room_walls
    clear_wall_sensor_cache()


def add_wall(x1, y1, x2, y2, thickness=None):
    wall = clamp_wall_to_arena(make_wall(x1, y1, x2, y2, thickness, kind="custom"))
    if distance((wall["x1"], wall["y1"]), (wall["x2"], wall["y2"])) < 1.0:
        print("벽의 시작점과 끝점이 너무 가깝다. 벽을 추가하지 않았다.")
        return False
    WALLS.append(wall)
    clear_wall_sensor_cache()
    print(
        f"벽 추가 완료: ({wall['x1']:.1f}, {wall['y1']:.1f}) -> "
        f"({wall['x2']:.1f}, {wall['y2']:.1f}), 두께 {wall['thickness']:.1f}"
    )
    return True


def list_walls():
    if not WALLS:
        print("현재 등록된 벽이 없다.")
        return
    print("\n===== 벽 목록 =====")
    for i, wall in enumerate(WALLS, start=1):
        print(
            f"{i}. [{wall.get('kind', 'custom')}] "
            f"({wall['x1']:.1f}, {wall['y1']:.1f}) -> "
            f"({wall['x2']:.1f}, {wall['y2']:.1f}), 두께 {wall['thickness']:.1f}"
        )


def clear_custom_walls():
    global WALLS
    WALLS = [wall for wall in WALLS if wall.get("kind") == "room"]
    clear_wall_sensor_cache()
    print("사용자가 추가한 벽을 모두 삭제했다. 방 벽은 유지된다.")


def remove_wall(index):
    if index < 1 or index > len(WALLS):
        print("삭제할 벽 번호가 올바르지 않다.")
        return False
    if WALLS[index - 1].get("kind") == "room":
        print("방을 구성하는 기본 벽은 삭제하지 않는다.")
        return False
    removed = WALLS.pop(index - 1)
    clear_wall_sensor_cache()
    print(
        f"벽 삭제 완료: ({removed['x1']:.1f}, {removed['y1']:.1f}) -> "
        f"({removed['x2']:.1f}, {removed['y2']:.1f})"
    )
    return True


def is_point_inside_wall(point, wall, extra_margin=0.0):
    a, b = wall_points(wall)
    return distance_point_to_segment(point, a, b) <= wall["thickness"] / 2.0 + extra_margin


def is_blocked_by_any_wall(path_start, path_end, extra_margin=0.0):
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


# -----------------------------
# 상자
# -----------------------------
def reset_box_position():
    global BOX_POS
    BOX_POS = BOX_START_POS
    # 매 판마다 상자가 기본 위치로 돌아가더라도 캐시는 지우지 않는다.
    # 캐시 키에 상자 위치가 포함되어 있으므로 같은 상자 위치의 캐시를 재사용할 수 있다.


def box_rect(pos=None):
    if pos is None:
        pos = BOX_POS
    half = BOX_SIZE / 2.0
    return {
        "left": pos[0] - half,
        "right": pos[0] + half,
        "bottom": pos[1] - half,
        "top": pos[1] + half,
    }


def point_inside_box(point, pos=None, margin=0.0):
    if not BOX_ENABLED:
        return False
    rect = box_rect(pos)
    return (
        rect["left"] - margin <= point[0] <= rect["right"] + margin
        and rect["bottom"] - margin <= point[1] <= rect["top"] + margin
    )


def box_overlaps_any_agent(box_pos, agent_positions, margin=None):
    """
    상자가 술래나 도망자의 몸과 겹치는지 확인한다.

    기존 문제:
    도망자가 상자를 밀었을 때 상자가 술래 위치를 덮어버리면
    술래가 상자 안에 갇히는 문제가 있었다.

    해결:
    상자의 새 위치가 어떤 AI의 위치와도 겹치면 상자 이동을 허용하지 않는다.
    """
    if margin is None:
        margin = AGENT_COLLISION_RADIUS

    for agent_pos in agent_positions:
        if agent_pos is None:
            continue

        if point_inside_box(agent_pos, box_pos, margin=margin):
            return True

    return False


def segment_intersects_box(p1, p2, pos=None, margin=0.0):
    if not BOX_ENABLED:
        return False
    rect = box_rect(pos)
    left = rect["left"] - margin
    right = rect["right"] + margin
    bottom = rect["bottom"] - margin
    top = rect["top"] + margin

    if left <= p1[0] <= right and bottom <= p1[1] <= top:
        return True
    if left <= p2[0] <= right and bottom <= p2[1] <= top:
        return True

    corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    for e1, e2 in edges:
        if segments_intersect(p1, p2, e1, e2):
            return True
    return False


def box_overlaps_wall(pos):
    if not BOX_ENABLED:
        return False
    rect = box_rect(pos)
    corners = [
        (rect["left"], rect["bottom"]),
        (rect["right"], rect["bottom"]),
        (rect["right"], rect["top"]),
        (rect["left"], rect["top"]),
    ]
    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    for wall in WALLS:
        wall_start, wall_end = wall_points(wall)
        margin = wall["thickness"] / 2.0 + BOX_PUSH_MARGIN
        for edge_start, edge_end in edges:
            if distance_segment_to_segment(edge_start, edge_end, wall_start, wall_end) <= margin:
                return True
        for corner in corners:
            if distance_point_to_segment(corner, wall_start, wall_end) <= margin:
                return True
    return False


def box_inside_arena(pos):
    rect = box_rect(pos)
    return (
        rect["left"] >= 0.0
        and rect["right"] <= ARENA_SIZE
        and rect["bottom"] >= 0.0
        and rect["top"] <= ARENA_SIZE
    )


def can_place_box(pos, forbidden_agent_positions=None):
    """
    상자를 특정 위치에 놓을 수 있는지 확인한다.

    forbidden_agent_positions:
    상자가 겹치면 안 되는 AI 위치들.
    예를 들어 도망자가 상자를 밀 때 술래 위치가 여기에 들어가면,
    상자가 술래를 덮어버리는 상황을 막을 수 있다.
    """
    if not BOX_ENABLED:
        return True

    if not box_inside_arena(pos):
        return False

    if box_overlaps_wall(pos):
        return False

    if forbidden_agent_positions is not None:
        if box_overlaps_any_agent(pos, forbidden_agent_positions):
            return False

    return True


def is_blocked_by_box(path_start, path_end):
    return segment_intersects_box(path_start, path_end, margin=BOX_PUSH_MARGIN)


def is_line_of_sight_blocked(path_start, path_end):
    return is_blocked_by_any_wall(path_start, path_end) or is_blocked_by_box(path_start, path_end)


def can_capture(chaser_pos, runner_pos):
    if distance(chaser_pos, runner_pos) > CAPTURE_RADIUS:
        return False
    if is_line_of_sight_blocked(chaser_pos, runner_pos):
        return False
    return True


def is_inside_room(pos):
    """
    좌표가 방 내부에 있는지 확인한다.
    방은 경기장 상단 중앙에 있는 250 x 250 영역이다.
    """
    return ROOM_LEFT <= pos[0] <= ROOM_RIGHT and ROOM_BOTTOM <= pos[1] <= ROOM_TOP


def get_door_centers():
    """왼쪽 문과 오른쪽 문의 중심 좌표를 반환한다."""
    return {
        "left": (ROOM_LEFT, DOOR_CENTER_Y),
        "right": (ROOM_RIGHT, DOOR_CENTER_Y),
    }


def box_to_nearest_door_info(pos=None):
    """
    상자 중심이 가장 가까운 출입구까지 얼마나 떨어져 있는지 계산한다.

    반환값:
    - door_name: "left" 또는 "right"
    - door_center: 가장 가까운 문 중심 좌표
    - dist: 상자 중심과 문 중심 사이 거리
    """
    if pos is None:
        pos = BOX_POS

    centers = get_door_centers()
    best_name = None
    best_center = None
    best_dist = float("inf")

    for name, center in centers.items():
        d = distance(pos, center)
        if d < best_dist:
            best_name = name
            best_center = center
            best_dist = d

    return best_name, best_center, best_dist


def door_blocked_by_box(door_name):
    """
    특정 출입구가 상자에 의해 막혀 있는지 확인한다.

    문은 세로 선분으로 보고,
    상자 사각형이 이 문 선분과 충분히 겹치면 막힌 것으로 본다.
    """
    centers = get_door_centers()

    if door_name not in centers:
        return False

    door_x, _ = centers[door_name]
    door_start = (door_x, DOOR_BOTTOM)
    door_end = (door_x, DOOR_TOP)

    return segment_intersects_box(door_start, door_end, BOX_POS, margin=0.0)


def get_door_block_status():
    """
    현재 상자가 어느 출입구를 막고 있는지 상태값으로 반환한다.

    0: 안 막힘
    1: 왼쪽 문 막힘
    2: 오른쪽 문 막힘
    3: 양쪽 문 막힘, 이론상 거의 없음
    """
    status = 0

    if door_blocked_by_box("left"):
        status += 1

    if door_blocked_by_box("right"):
        status += 2

    return status


def door_block_status_text(status=None):
    if status is None:
        status = get_door_block_status()

    if status == 0:
        return "none"
    if status == 1:
        return "left"
    if status == 2:
        return "right"
    return "both"


def box_blocks_between_agents(chaser_pos, runner_pos, box_pos=None):
    """
    상자가 술래와 도망자 사이의 직선 경로를 실제로 가로막는지 확인한다.

    True이면:
    술래와 도망자를 잇는 선분이 상자 사각형과 교차한다는 뜻이다.
    """
    if box_pos is None:
        box_pos = BOX_POS

    return segment_intersects_box(chaser_pos, runner_pos, box_pos, margin=0.0)


def box_interference_score(chaser_pos, runner_pos, box_pos=None):
    """
    상자가 술래와 도망자 사이를 얼마나 잘 방해하고 있는지 0~1 점수로 계산한다.

    1에 가까울수록:
    - 상자가 술래와 도망자 사이의 직선 경로에 가깝다.
    - 즉, 술래 입장에서 도망자에게 가는 길을 상자가 방해할 가능성이 크다.

    0에 가까울수록:
    - 상자가 두 AI의 추격 경로와 별 상관없는 곳에 있다.
    """
    if box_pos is None:
        box_pos = BOX_POS

    if box_blocks_between_agents(chaser_pos, runner_pos, box_pos):
        return 1.0

    line_distance = distance_point_to_segment(box_pos, chaser_pos, runner_pos)

    # 상자 중심이 술래-도망자 연결선에서 이 거리 이상 멀면 방해 효과가 거의 없다고 본다.
    effective_distance = 160.0

    score = 1.0 - line_distance / effective_distance
    return clamp(score, 0.0, 1.0)


def box_interference_status_text(chaser_pos, runner_pos, box_pos=None):
    if box_blocks_between_agents(chaser_pos, runner_pos, box_pos):
        return "blocking"
    score = box_interference_score(chaser_pos, runner_pos, box_pos)

    if score >= 0.7:
        return "near-path"
    if score >= 0.35:
        return "partial"
    return "none"


# -----------------------------
# 상태 / 이동
# -----------------------------
def random_start_positions():
    while True:
        chaser_pos = (
            random.uniform(WALL_CLEARANCE, ARENA_SIZE - WALL_CLEARANCE),
            random.uniform(WALL_CLEARANCE, ARENA_SIZE - WALL_CLEARANCE),
        )
        runner_pos = (
            random.uniform(WALL_CLEARANCE, ARENA_SIZE - WALL_CLEARANCE),
            random.uniform(WALL_CLEARANCE, ARENA_SIZE - WALL_CLEARANCE),
        )
        invalid = False
        for pos in (chaser_pos, runner_pos):
            if any(is_point_inside_wall(pos, wall, extra_margin=WALL_CLEARANCE) for wall in WALLS):
                invalid = True
            if point_inside_box(pos, margin=10.0):
                invalid = True
        if (not invalid) and distance(chaser_pos, runner_pos) >= MIN_START_DISTANCE:
            return chaser_pos, runner_pos


def get_start_positions():
    if START_MODE == "fixed":
        return FIXED_CHASER_POS, FIXED_RUNNER_POS
    return random_start_positions()


def angle_to_bin(dx, dy):
    angle = math.atan2(dy, dx)
    normalized = (angle + math.pi) / (2 * math.pi)
    return int(normalized * ANGLE_BINS) % ANGLE_BINS


def value_to_bin(value, low, high, bins):
    value = clamp(value, low, high)
    ratio = (value - low) / (high - low)
    index = int(ratio * bins)
    if index >= bins:
        index = bins - 1
    return index


def wall_zone(value):
    if value < WALL_MARGIN:
        return 0
    if value > ARENA_SIZE - WALL_MARGIN:
        return 2
    return 1


def get_wall_sensor_cache_key(position, speed):
    qx = int(position[0] // WALL_SENSOR_CACHE_GRID)
    qy = int(position[1] // WALL_SENSOR_CACHE_GRID)
    box_qx = int(BOX_POS[0] // WALL_SENSOR_CACHE_GRID)
    box_qy = int(BOX_POS[1] // WALL_SENSOR_CACHE_GRID)
    return qx, qy, box_qx, box_qy, round(speed, 3), len(WALLS)


def get_blocked_action_mask(position, speed):
    key = get_wall_sensor_cache_key(position, speed)
    if key in WALL_SENSOR_CACHE:
        return WALL_SENSOR_CACHE[key]
    blocked = []
    for dx, dy in ACTIONS:
        x, y = position
        raw_x = x + dx * speed * WALL_SENSOR_LOOKAHEAD
        raw_y = y + dy * speed * WALL_SENSOR_LOOKAHEAD
        outside = raw_x < 0.0 or raw_x > ARENA_SIZE or raw_y < 0.0 or raw_y > ARENA_SIZE
        if outside:
            blocked.append(1)
            continue
        candidate = (raw_x, raw_y)
        blocked_by_wall = is_blocked_by_any_wall(position, candidate)
        blocked_by_box = is_blocked_by_box(position, candidate)
        blocked.append(1 if blocked_by_wall or blocked_by_box else 0)
    result = tuple(blocked)
    WALL_SENSOR_CACHE[key] = result
    return result


def make_agent_state(self_pos, opponent_pos, self_speed):
    dx = opponent_pos[0] - self_pos[0]
    dy = opponent_pos[1] - self_pos[1]
    angle_bin = angle_to_bin(dx, dy)
    dist_bin = value_to_bin(distance(self_pos, opponent_pos), 0.0, math.sqrt(2) * ARENA_SIZE, DISTANCE_BINS)

    x_wall = wall_zone(self_pos[0])
    y_wall = wall_zone(self_pos[1])
    blocked_actions = get_blocked_action_mask(self_pos, self_speed)

    # 상자 위치 정보
    box_dx = BOX_POS[0] - self_pos[0]
    box_dy = BOX_POS[1] - self_pos[1]
    box_angle_bin = angle_to_bin(box_dx, box_dy)
    box_dist_bin = value_to_bin(distance(self_pos, BOX_POS), 0.0, math.sqrt(2) * ARENA_SIZE, DISTANCE_BINS)

    # 상자가 술래와 도망자 사이를 얼마나 방해하고 있는지 상태에 포함한다.
    # 이 정보 덕분에 도망자는 단순히 문 좌표를 외우는 것이 아니라,
    # 술래의 추격 경로를 상자로 막는 상황을 학습할 수 있다.
    interference_score = box_interference_score(
        self_pos if self_speed == RUNNER_SPEED else opponent_pos,
        opponent_pos if self_speed == RUNNER_SPEED else self_pos,
        BOX_POS,
    )
    interference_bin = value_to_bin(interference_score, 0.0, 1.0, 4)

    box_blocks_path = 1 if box_blocks_between_agents(
        self_pos if self_speed == RUNNER_SPEED else opponent_pos,
        opponent_pos if self_speed == RUNNER_SPEED else self_pos,
        BOX_POS,
    ) else 0

    # 방 안/밖 상태는 여전히 전략 판단에 도움이 되므로 유지한다.
    self_in_room = 1 if is_inside_room(self_pos) else 0
    opponent_in_room = 1 if is_inside_room(opponent_pos) else 0

    return (
        angle_bin,
        dist_bin,
        x_wall,
        y_wall,
        blocked_actions,
        box_angle_bin,
        box_dist_bin,
        interference_bin,
        box_blocks_path,
        self_in_room,
        opponent_in_room,
    )

def get_epsilon(total_episodes_done):
    progress = min(total_episodes_done / EPSILON_DECAY_EPISODES, 1.0)
    return EPSILON_START * (1 - progress) + EPSILON_END * progress


def path_grid_count():
    return int(ARENA_SIZE // PATH_GRID_SIZE)


def pos_to_path_cell(pos):
    """
    연속 좌표를 A* 길찾기용 격자 좌표로 변환한다.
    실제 게임 좌표가 격자로 바뀌는 것은 아니고,
    길찾기 계산을 위해서만 임시로 사용한다.
    """
    n = path_grid_count()
    cx = int(round(pos[0] / PATH_GRID_SIZE))
    cy = int(round(pos[1] / PATH_GRID_SIZE))
    cx = max(0, min(n, cx))
    cy = max(0, min(n, cy))
    return cx, cy


def path_cell_to_pos(cell):
    x = clamp(cell[0] * PATH_GRID_SIZE, 0.0, ARENA_SIZE)
    y = clamp(cell[1] * PATH_GRID_SIZE, 0.0, ARENA_SIZE)
    return x, y


def path_cell_walkable(cell):
    """
    A* 길찾기에서 해당 격자점이 지나갈 수 있는 위치인지 판단한다.
    벽, 상자, 경기장 밖은 지나갈 수 없다.
    """
    pos = path_cell_to_pos(cell)

    if not inside_arena_with_clearance(pos):
        return False

    if any(is_point_inside_wall(pos, wall, extra_margin=WALL_CLEARANCE) for wall in WALLS):
        return False

    if point_inside_box(pos, BOX_POS, margin=AGENT_COLLISION_RADIUS):
        return False

    return True


def path_edge_walkable(cell_a, cell_b):
    """
    A*에서 한 격자점에서 다음 격자점으로 이동 가능한지 판단한다.
    중간 경로가 벽이나 상자를 통과하면 불가능하다.
    """
    pos_a = path_cell_to_pos(cell_a)
    pos_b = path_cell_to_pos(cell_b)

    if is_blocked_by_any_wall(pos_a, pos_b, extra_margin=WALL_CLEARANCE):
        return False

    if segment_intersects_box(pos_a, pos_b, BOX_POS, margin=AGENT_COLLISION_RADIUS):
        return False

    return True


def path_heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_walkable_cell(cell, max_radius=3):
    """
    목표 격자점이 벽/상자 때문에 막혀 있을 때,
    근처의 지나갈 수 있는 격자점을 찾는다.
    """
    if path_cell_walkable(cell):
        return cell

    cx, cy = cell
    n = path_grid_count()

    for radius in range(1, max_radius + 1):
        candidates = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue

                nx = cx + dx
                ny = cy + dy

                if 0 <= nx <= n and 0 <= ny <= n:
                    candidates.append((nx, ny))

        candidates.sort(key=lambda c: path_heuristic(cell, c))

        for candidate in candidates:
            if path_cell_walkable(candidate):
                return candidate

    return None


def astar_next_waypoint(start_pos, goal_pos):
    """
    술래가 도망자에게 가기 위한 다음 waypoint를 A*로 계산한다.
    벽과 상자를 피해서 이동하는 경로를 찾는다.
    """
    start_cell = pos_to_path_cell(start_pos)
    goal_cell = pos_to_path_cell(goal_pos)

    box_cell = pos_to_path_cell(BOX_POS)
    cache_key = (
        start_cell,
        goal_cell,
        box_cell,
        len(WALLS),
        round(PATH_GRID_SIZE, 3),
    )

    if cache_key in PATHFINDING_CACHE:
        cached = PATHFINDING_CACHE[cache_key]
        if cached is None:
            return None
        return path_cell_to_pos(cached)

    start_cell = nearest_walkable_cell(start_cell, max_radius=2)
    goal_cell = nearest_walkable_cell(goal_cell, max_radius=3)

    if start_cell is None or goal_cell is None:
        PATHFINDING_CACHE[cache_key] = None
        return None

    if start_cell == goal_cell:
        PATHFINDING_CACHE[cache_key] = goal_cell
        return goal_pos

    neighbor_dirs = [
        (0, 1), (0, -1), (-1, 0), (1, 0),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
    ]

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_cell))

    came_from = {}
    g_score = {start_cell: 0.0}
    closed = set()
    n = path_grid_count()
    max_expansions = 1200
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        expansions += 1

        if current == goal_cell:
            break

        for dx, dy in neighbor_dirs:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] <= n and 0 <= neighbor[1] <= n):
                continue

            if neighbor in closed:
                continue

            if not path_cell_walkable(neighbor):
                continue

            if not path_edge_walkable(current, neighbor):
                continue

            move_cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + path_heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f_score, neighbor))

    if goal_cell not in came_from and goal_cell != start_cell:
        PATHFINDING_CACHE[cache_key] = None
        return None

    path = [goal_cell]
    current = goal_cell

    while current != start_cell:
        current = came_from[current]
        path.append(current)

    path.reverse()

    if len(path) < 2:
        next_cell = goal_cell
    else:
        next_cell = path[1]

    PATHFINDING_CACHE[cache_key] = next_cell
    return path_cell_to_pos(next_cell)


def action_is_clear_without_pushing_box(position, action_index, speed):
    """
    길찾기용 행동 유효성 검사.
    이 함수는 실제 BOX_POS를 변경하지 않는다.
    술래가 상자를 밀기보다 우회 경로를 찾도록 하기 위해,
    상자를 장애물로 보고 피하게 한다.
    """
    dx, dy = ACTIONS[action_index]
    raw_x = position[0] + dx * speed
    raw_y = position[1] + dy * speed

    candidate = (raw_x, raw_y)

    if not inside_arena_with_clearance(candidate):
        return False

    if wall_collision_for_agent(position, candidate):
        return False

    if segment_intersects_box(position, candidate, BOX_POS, margin=AGENT_COLLISION_RADIUS):
        return False

    if point_inside_box(candidate, BOX_POS, margin=AGENT_COLLISION_RADIUS):
        return False

    return True


def direction_to_action_with_obstacle_check(position, target_pos, speed):
    """
    target_pos 방향으로 가는 8방향 행동 중 가장 좋은 행동을 고른다.
    단, 벽/상자에 바로 막히는 행동은 제외한다.
    """
    vx = target_pos[0] - position[0]
    vy = target_pos[1] - position[1]
    length = math.hypot(vx, vy)

    if length < 1e-9:
        return None

    vx /= length
    vy /= length

    scored_actions = []

    for i, (dx, dy) in enumerate(ACTIONS):
        alignment = dx * vx + dy * vy

        # 행동 후 목표와의 거리가 얼마나 줄어드는지도 함께 반영
        candidate = (position[0] + dx * speed, position[1] + dy * speed)
        distance_gain = distance(position, target_pos) - distance(candidate, target_pos)

        score = alignment + 0.05 * distance_gain
        scored_actions.append((score, i))

    scored_actions.sort(reverse=True)

    for _, action_index in scored_actions:
        if action_is_clear_without_pushing_box(position, action_index, speed):
            return action_index

    return None


def reset_chaser_stuck_history():
    """
    매 판 시작 시 술래의 제자리걸음/헛돎 감지 기록을 초기화한다.
    """
    CHASER_STUCK_HISTORY.clear()


def update_chaser_stuck_history(chaser_pos, runner_pos):
    """
    술래의 최근 위치와 술래-도망자 거리를 기록한다.
    """
    if not CHASER_STUCK_DETECTION_ENABLED:
        return

    current_distance = distance(chaser_pos, runner_pos)

    CHASER_STUCK_HISTORY.append({
        "pos": chaser_pos,
        "distance": current_distance,
    })

    max_len = max(3, CHASER_STUCK_WINDOW_STEPS + 2)

    if len(CHASER_STUCK_HISTORY) > max_len:
        del CHASER_STUCK_HISTORY[0:len(CHASER_STUCK_HISTORY) - max_len]


def chaser_position_cell(pos):
    """
    술래가 비슷한 위치를 반복하는지 보기 위해 좌표를 거친 구역으로 압축한다.
    """
    return (
        int(pos[0] // CHASER_STUCK_POSITION_GRID),
        int(pos[1] // CHASER_STUCK_POSITION_GRID),
    )


def is_chaser_strategically_stuck(chaser_pos, runner_pos):
    """
    술래가 물리적으로 완전히 갇힌 것은 아니지만,
    전략적으로 헛도는 상황인지 판단한다.

    감지하는 상황:
    1. 최근 몇 step 동안 도망자와의 거리를 거의 줄이지 못함
    2. 최근 몇 step 동안 비슷한 위치 구역을 왔다갔다 함
    3. 도망자와 충분히 가까운 상황이 아니라서 직접 추격만으로 해결되기 어려움
    """
    if not CHASER_STUCK_DETECTION_ENABLED:
        return False

    current_distance = distance(chaser_pos, runner_pos)

    # 이미 충분히 가까우면 A*를 쓰지 않고 직접 추격하는 편이 낫다.
    if current_distance <= CHASER_STUCK_IGNORE_DISTANCE:
        return False

    if len(CHASER_STUCK_HISTORY) < CHASER_STUCK_WINDOW_STEPS:
        return False

    recent = CHASER_STUCK_HISTORY[-CHASER_STUCK_WINDOW_STEPS:]
    old_distance = recent[0]["distance"]
    recent_progress = old_distance - current_distance

    # 거리를 충분히 줄이지 못했는지
    poor_progress = recent_progress < CHASER_STUCK_MIN_PROGRESS

    # 위치가 비슷한 구역에서 반복되는지
    cells = [chaser_position_cell(item["pos"]) for item in recent]
    unique_cells = set(cells)
    oscillating = len(unique_cells) <= max(2, CHASER_STUCK_WINDOW_STEPS // 3)

    # 2-step 왕복 패턴 감지: A-B-A-B 비슷한 상황
    back_and_forth = False
    if len(cells) >= 6:
        same_as_two_steps_ago = 0
        for i in range(2, len(cells)):
            if cells[i] == cells[i - 2]:
                same_as_two_steps_ago += 1
        back_and_forth = same_as_two_steps_ago >= len(cells) // 2

    return poor_progress and (oscillating or back_and_forth)


def inside_arena_with_clearance(pos, margin=None):
    """
    AI 중심이 경기장 외곽선에 너무 붙지 않도록 검사한다.
    """
    if margin is None:
        margin = WALL_CLEARANCE

    return (
        margin <= pos[0] <= ARENA_SIZE - margin
        and margin <= pos[1] <= ARENA_SIZE - margin
    )


def clamp_position_with_clearance(pos, margin=None):
    """
    AI 중심이 외곽선 위에 올라타지 않도록 좌표를 보정한다.
    """
    if margin is None:
        margin = WALL_CLEARANCE

    return (
        clamp(pos[0], margin, ARENA_SIZE - margin),
        clamp(pos[1], margin, ARENA_SIZE - margin),
    )


def wall_collision_for_agent(path_start, path_end):
    """
    AI를 점이 아니라 작은 원으로 보고 벽 충돌을 검사한다.
    """
    return is_blocked_by_any_wall(path_start, path_end, extra_margin=WALL_CLEARANCE)


def choose_room_entry_side(chaser_pos, runner_pos):
    """
    방 안에 있는 도망자를 쫓을 때 왼쪽 문/오른쪽 문 중 어느 쪽으로 돌아갈지 고른다.
    전체 A*를 매번 돌리지 않고, 방 구조를 이용한 간단한 경로 길이 추정만 한다.
    """
    sides = []

    for side_name, sign, wall_x in [
        ("left", -1.0, ROOM_LEFT),
        ("right", 1.0, ROOM_RIGHT),
    ]:
        outside_x = wall_x + sign * DOOR_WAYPOINT_OFFSET
        inside_x = wall_x - sign * DOOR_WAYPOINT_OFFSET

        bottom_corner = (outside_x, ROOM_BOTTOM - DOOR_WAYPOINT_OFFSET)
        door_outer = (outside_x, DOOR_CENTER_Y)
        door_inner = (inside_x, DOOR_CENTER_Y)

        estimated = (
            distance(chaser_pos, bottom_corner)
            + distance(bottom_corner, door_outer)
            + distance(door_outer, door_inner)
            + distance(door_inner, runner_pos)
        )

        sides.append((estimated, side_name, bottom_corner, door_outer, door_inner))

    sides.sort(key=lambda item: item[0])
    return sides[0]


def get_chaser_room_waypoint(chaser_pos, runner_pos):
    """
    술래가 방 안의 도망자를 쫓을 때 방 벽 앞에서 헛돌지 않도록,
    문으로 돌아가는 중간 목표점을 준다.

    반환값:
    - None: 별도 waypoint 필요 없음
    - 좌표: 우선 향해야 할 waypoint
    """
    if not CHASER_DOOR_WAYPOINT_ENABLED:
        return None

    # 도망자가 방 안에 있을 때만 적용한다.
    if not is_inside_room(runner_pos):
        return None

    # 술래가 이미 방 안에 있으면 그냥 도망자 추격.
    if is_inside_room(chaser_pos):
        return None

    _, side_name, bottom_corner, door_outer, door_inner = choose_room_entry_side(chaser_pos, runner_pos)

    # 방 아래쪽 벽 바로 밑 중앙 부근에 있으면 먼저 좌/우 아래 코너로 빠져야 한다.
    below_room = chaser_pos[1] < ROOM_BOTTOM
    horizontally_under_room = ROOM_LEFT - DOOR_WAYPOINT_OFFSET < chaser_pos[0] < ROOM_RIGHT + DOOR_WAYPOINT_OFFSET

    if below_room and horizontally_under_room:
        if distance(chaser_pos, bottom_corner) > DOOR_WAYPOINT_REACH_RADIUS:
            return bottom_corner

    # 방의 바깥쪽 측면에 있으면 문 바깥쪽 중심으로 이동.
    if distance(chaser_pos, door_outer) > DOOR_WAYPOINT_REACH_RADIUS:
        return door_outer

    # 문 앞에 도착했으면 방 안쪽으로 들어간다.
    if distance(chaser_pos, door_inner) > DOOR_WAYPOINT_REACH_RADIUS:
        return door_inner

    return None


def chaser_needs_forced_door_route(chaser_pos, runner_pos):
    """
    술래가 방 아래 벽 앞에서 헛도는 것을 막기 위해
    출입구 우회 경로를 강제로 적용해야 하는지 판단한다.
    """
    if not CHASER_FORCE_DOOR_ROUTE:
        return False

    if not CHASER_DOOR_WAYPOINT_ENABLED:
        return False

    # 도망자가 방 안에 있고 술래가 방 밖에 있을 때만 강제 우회.
    if not is_inside_room(runner_pos):
        return False

    if is_inside_room(chaser_pos):
        return False

    return True


def get_forced_door_route_target(chaser_pos, runner_pos):
    """
    방 안의 도망자를 쫓을 때 술래의 강제 목표점을 반환한다.

    기존 get_chaser_room_waypoint보다 더 강한 규칙:
    - 술래가 방 아래 벽 근처 중앙에 있으면 무조건 좌/우 문 바깥쪽으로 이동
    - 문 바깥쪽에 도착하면 문 안쪽으로 진입
    - 방 안에 들어가면 강제 우회 해제
    """
    if not chaser_needs_forced_door_route(chaser_pos, runner_pos):
        return None

    _, side_name, bottom_corner, door_outer, door_inner = choose_room_entry_side(chaser_pos, runner_pos)

    below_room = chaser_pos[1] < ROOM_BOTTOM
    near_bottom_wall = ROOM_BOTTOM - ROOM_WALL_AVOID_BAND <= chaser_pos[1] <= ROOM_BOTTOM + 5.0
    under_room_x_range = ROOM_LEFT - DOOR_WAYPOINT_OFFSET < chaser_pos[0] < ROOM_RIGHT + DOOR_WAYPOINT_OFFSET

    # 방 아래 벽 바로 앞 중앙에 있으면, 벽에서 떨어지며 좌/우 측면으로 빠진다.
    if below_room and near_bottom_wall and under_room_x_range:
        return bottom_corner

    # 아직 문 바깥쪽에 충분히 가까이 가지 않았으면 문 바깥쪽 중심으로 이동.
    if distance(chaser_pos, door_outer) > DOOR_WAYPOINT_REACH_RADIUS:
        return door_outer

    # 문 바깥쪽에 도착하면 문 안쪽으로 들어간다.
    return door_inner


def get_forced_door_route_action(chaser_pos, runner_pos):
    """
    강제 출입구 우회 경로에 따른 행동을 반환한다.
    반환값: action_index 또는 None
    """
    target = get_forced_door_route_target(chaser_pos, runner_pos)

    if target is None:
        return None

    action = direction_to_action_with_obstacle_check(chaser_pos, target, CHASER_SPEED)

    if action is not None:
        return action

    # direction_to_action_with_obstacle_check가 너무 보수적으로 실패하면,
    # map-aware 후보 중 target에 가장 가까워지는 행동을 직접 고른다.
    valid_actions = get_valid_actions_map_aware(
        chaser_pos,
        CHASER_SPEED,
        other_agent_pos=runner_pos,
        allow_box_push=CHASER_CAN_PUSH_BOX,
    )

    if not valid_actions:
        return None

    best_action = None
    best_distance = float("inf")

    for action_index in valid_actions:
        dx, dy = ACTIONS[action_index]
        candidate = (
            chaser_pos[0] + dx * CHASER_SPEED,
            chaser_pos[1] + dy * CHASER_SPEED,
        )

        d = distance(candidate, target)

        if d < best_distance:
            best_distance = d
            best_action = action_index

    return best_action


def can_take_action_map_aware(position, action_index, speed, other_agent_pos=None, allow_box_push=True):
    """
    AI가 해당 행동을 실제로 할 수 있는지 미리 판단한다.

    핵심:
    - 벽에 부딪혀 본 뒤 배우는 것이 아니라,
    - 현재 맵 구조를 보고 막힌 방향을 사전에 제외한다.

    검사 항목:
    1. 경기장 밖으로 나가는가?
    2. 이동 경로가 벽을 통과하는가?
    3. 상자와 부딪히는가?
       - 상자를 밀 수 있으면, 상자의 새 위치가 가능한지도 검사
       - 상자를 밀 수 없으면 해당 행동은 불가능
    4. 이동 후 AI가 상자 안에 들어가는가?
    """
    if action_index is None:
        return False

    dx, dy = ACTIONS[action_index]

    raw_x = position[0] + dx * speed
    raw_y = position[1] + dy * speed

    candidate_pos = (raw_x, raw_y)

    # 경기장 외곽선 위까지 가지 않고 AI 반경만큼 여유를 둔다.
    if not inside_arena_with_clearance(candidate_pos):
        return False

    # 벽도 AI 반경만큼 부풀려서 검사한다.
    if wall_collision_for_agent(position, candidate_pos):
        return False

    # 상자와 부딪히는 행동인지 확인
    if BOX_ENABLED and segment_intersects_box(position, candidate_pos, margin=BOX_PUSH_MARGIN):
        if not allow_box_push:
            return False

        candidate_box_pos = (
            BOX_POS[0] + dx * speed,
            BOX_POS[1] + dy * speed,
        )

        forbidden_positions = [other_agent_pos, candidate_pos]

        if not can_place_box(candidate_box_pos, forbidden_agent_positions=forbidden_positions):
            return False

        if point_inside_box(candidate_pos, candidate_box_pos, margin=AGENT_COLLISION_RADIUS):
            return False

        return True

    # 상자를 직접 밀지 않는 일반 이동에서도 상자 안으로 들어가면 안 된다.
    if point_inside_box(candidate_pos, BOX_POS, margin=AGENT_COLLISION_RADIUS):
        return False

    return True


def get_valid_actions_map_aware(position, speed, other_agent_pos=None, allow_box_push=True):
    """
    현재 위치에서 실제로 선택 가능한 행동 번호 목록을 반환한다.
    """
    if not MAP_AWARE_ACTION_MASKING:
        return list(range(len(ACTIONS)))

    valid_actions = []

    for action_index in range(len(ACTIONS)):
        if can_take_action_map_aware(
            position,
            action_index,
            speed,
            other_agent_pos=other_agent_pos,
            allow_box_push=allow_box_push,
        ):
            valid_actions.append(action_index)

    return valid_actions


def choose_action_from_q_with_mask(agent, state, epsilon, valid_actions):
    """
    Q-table에서 행동을 고르되, 막힌 방향은 선택하지 않는다.

    기존 방식:
    - 8방향 전체 중 선택
    - 벽/상자 방향도 선택될 수 있음

    수정 방식:
    - valid_actions 안에서만 선택
    - 랜덤 탐험도 valid_actions 안에서만 수행
    """
    if not valid_actions:
        # 정말 선택 가능한 행동이 없을 때만 기존 방식으로 fallback.
        # 실제로는 거의 발생하지 않아야 한다.
        return agent.choose_action(state, epsilon)

    if MAP_AWARE_EXPLORATION and random.random() < epsilon:
        return random.choice(valid_actions)

    q_values = agent.q_table[state]
    best_q = max(q_values[action_index] for action_index in valid_actions)
    best_actions = [action_index for action_index in valid_actions if q_values[action_index] == best_q]
    return random.choice(best_actions)


def get_chaser_fast_greedy_action(chaser_pos, runner_pos):
    """
    A* 없이 매우 빠르게 술래 행동을 고르는 함수.

    - 8방향 후보만 검사한다.
    - 벽/상자에 바로 막히는 행동은 제외한다.
    - 남은 행동 중 도망자와 가장 가까워지는 행동을 선택한다.

    계산량은 거의 O(8)이라 기존 Q-table 방식과 큰 차이가 적다.
    """
    candidates = []

    forced_target = get_forced_door_route_target(chaser_pos, runner_pos)
    waypoint = forced_target if forced_target is not None else get_chaser_room_waypoint(chaser_pos, runner_pos)
    target_pos = waypoint if waypoint is not None else runner_pos

    current_distance = distance(chaser_pos, target_pos)

    for action_index, (dx, dy) in enumerate(ACTIONS):
        raw_x = chaser_pos[0] + dx * CHASER_SPEED
        raw_y = chaser_pos[1] + dy * CHASER_SPEED

        if raw_x < 0.0 or raw_x > ARENA_SIZE or raw_y < 0.0 or raw_y > ARENA_SIZE:
            continue

        candidate_pos = (raw_x, raw_y)

        if is_blocked_by_any_wall(chaser_pos, candidate_pos):
            continue

        if not can_take_action_map_aware(
            chaser_pos,
            action_index,
            CHASER_SPEED,
            other_agent_pos=runner_pos,
            allow_box_push=CHASER_CAN_PUSH_BOX,
        ):
            continue

        new_distance = distance(candidate_pos, runner_pos)
        distance_gain = current_distance - new_distance

        vx = runner_pos[0] - chaser_pos[0]
        vy = runner_pos[1] - chaser_pos[1]
        v_len = math.hypot(vx, vy)

        if v_len > 1e-9:
            vx /= v_len
            vy /= v_len
            alignment = dx * vx + dy * vy
        else:
            alignment = 0.0

        obstacle_penalty = 0.0
        if point_inside_box(candidate_pos, BOX_POS, margin=AGENT_COLLISION_RADIUS * 2.0):
            obstacle_penalty -= 0.4

        score = distance_gain * 1.0 + alignment * 2.0 + obstacle_penalty
        candidates.append((score, action_index))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def get_chaser_pathfinding_action(chaser_pos, runner_pos):
    """
    술래 전용 길찾기 행동 선택.

    fast 모드:
    - 기본은 8방향 빠른 장애물 회피 추격을 사용한다.
    - 물리적으로 완전히 막혔다는 이유만으로 A*를 돌리지 않는다.
    - 술래가 최근 몇 step 동안 제자리걸음/헛돎 상태로 감지될 때만 A*를 사용한다.

    astar 모드:
    - 기존처럼 A* 기반 우회 경로를 적극 사용한다.
    """
    if CHASER_PATH_MODE == "fast":
        stuck = is_chaser_strategically_stuck(chaser_pos, runner_pos)

        if stuck and CHASER_ASTAR_FALLBACK_ENABLED:
            # 헛돌고 있다고 판단될 때만 A*를 사용한다.
            # fallback_rate를 1보다 작게 두면 너무 자주 A*를 돌리는 것을 막을 수 있다.
            if random.random() < CHASER_ASTAR_FALLBACK_RATE:
                waypoint = astar_next_waypoint(chaser_pos, runner_pos)
                if waypoint is not None:
                    action = direction_to_action_with_obstacle_check(chaser_pos, waypoint, CHASER_SPEED)
                    if action is not None:
                        return action, "ASTAR_STUCK"

        fast_action = get_chaser_fast_greedy_action(chaser_pos, runner_pos)

        if fast_action is not None:
            return fast_action, "FASTPATH_STUCK" if stuck else "FASTPATH"

        # fast_action이 없더라도 여기서 바로 A*를 강제로 쓰지 않는다.
        # 사용자가 원한 기준은 물리적 막힘이 아니라 제자리걸음/헛돎이기 때문이다.
        return None, "NO_FAST_ACTION"

    # astar 모드: 기존 방식
    if not is_line_of_sight_blocked(chaser_pos, runner_pos):
        direct_action = direction_to_action_with_obstacle_check(chaser_pos, runner_pos, CHASER_SPEED)
        if direct_action is not None:
            return direct_action, "DIRECT"

    waypoint = astar_next_waypoint(chaser_pos, runner_pos)

    if waypoint is None:
        return None, "ASTAR_FAIL"

    action = direction_to_action_with_obstacle_check(chaser_pos, waypoint, CHASER_SPEED)
    return action, "ASTAR"




def choose_chaser_action(chaser_ai, chaser_state, chaser_pos, runner_pos, epsilon):
    """
    술래 행동 선택.

    강화 내용:
    - 도망자가 방 안에 있고 술래가 방 밖에 있으면,
      Q-table/탐험보다 먼저 출입구 우회 경로를 강제한다.
    - 이렇게 해야 방 아래 벽 앞에서 술래가 계속 들이받지 않는다.
    - 그 외 상황에서는 기존 Q-table, 빠른 추격, 헛돎 감지 A*를 사용한다.
    """
    valid_actions = get_valid_actions_map_aware(
        chaser_pos,
        CHASER_SPEED,
        other_agent_pos=runner_pos,
        allow_box_push=CHASER_CAN_PUSH_BOX,
    )

    # 최우선: 방 안 도망자를 쫓을 때는 문으로 돌아가는 행동을 강제한다.
    forced_door_action = get_forced_door_route_action(chaser_pos, runner_pos)

    if forced_door_action is not None and forced_door_action in valid_actions:
        return forced_door_action, "DOOR_ROUTE"

    # 학습 초반 탐험도 맵 구조를 알고 가능한 행동 안에서만 수행한다.
    if random.random() < epsilon:
        if valid_actions:
            return random.choice(valid_actions), "EXPLORE_MASKED"
        return chaser_ai.choose_action(chaser_state, epsilon=1.0), "EXPLORE"

    if CHASER_PATHFINDING_ENABLED and random.random() < CHASER_PATHFINDING_RATE:
        path_action, control = get_chaser_pathfinding_action(chaser_pos, runner_pos)

        if path_action is not None and path_action in valid_actions:
            return path_action, control

    # Q-table 행동도 가능한 방향 안에서만 선택한다.
    action = choose_action_from_q_with_mask(
        chaser_ai,
        chaser_state,
        epsilon=0.0,
        valid_actions=valid_actions,
    )
    return action, "Q_MASKED"




def move_position_with_box(position, action_index, speed, other_agent_pos=None):
    global BOX_POS

    # 만약 어떤 이유로 이미 AI가 상자 안에 들어가 있다면,
    # 더 안쪽으로 움직이는 것을 막고 현재 위치에 머물게 한다.
    if point_inside_box(position, BOX_POS, margin=0.0):
        return position, True, False, True

    x, y = position
    dx, dy = ACTIONS[action_index]
    raw_x = x + dx * speed
    raw_y = y + dy * speed

    candidate_pos = (raw_x, raw_y)

    if not inside_arena_with_clearance(candidate_pos):
        return position, True, False, False

    candidate_pos = clamp_position_with_clearance(candidate_pos)
    hit_arena_wall = False

    if wall_collision_for_agent(position, candidate_pos):
        return position, True, False, False

    pushed_box = False
    box_blocked = False
    if BOX_ENABLED and segment_intersects_box(position, candidate_pos, margin=BOX_PUSH_MARGIN):
        box_dx = dx * speed
        box_dy = dy * speed
        candidate_box_pos = (BOX_POS[0] + box_dx, BOX_POS[1] + box_dy)

        # 상자가 술래/도망자의 몸을 덮어버리는 것을 막는다.
        # other_agent_pos는 반대편 AI의 현재 위치다.
        # candidate_pos는 상자를 민 AI가 이번 step에서 도착하려는 위치다.
        forbidden_positions = [other_agent_pos, candidate_pos]

        if can_place_box(candidate_box_pos, forbidden_agent_positions=forbidden_positions):
            # 상자를 밀고 난 뒤에도 미는 AI가 상자 안에 들어가면 안 된다.
            if point_inside_box(candidate_pos, candidate_box_pos, margin=AGENT_COLLISION_RADIUS):
                box_blocked = True
                return position, True, False, True

            BOX_POS = candidate_box_pos
            # 캐시 키에 상자 위치가 포함되어 있으므로 상자가 움직였다고 매번 전체 캐시를 비우지 않는다.
            pushed_box = True
        else:
            box_blocked = True
            return position, True, False, True

    # 상자를 밀지 않는 일반 이동에서도 AI가 상자 안으로 들어가는 것을 막는다.
    if point_inside_box(candidate_pos, BOX_POS, margin=AGENT_COLLISION_RADIUS):
        return position, True, False, True

    return candidate_pos, hit_arena_wall, pushed_box, box_blocked


# -----------------------------
# 한 step / 한 episode
# -----------------------------
def simulate_one_step(chaser_ai, runner_ai, chaser_pos, runner_pos, epsilon, step_index, max_steps, training=True):
    old_chaser_pos = chaser_pos
    old_runner_pos = runner_pos
    old_distance = distance(chaser_pos, runner_pos)
    old_box_pos = BOX_POS
    old_box_interference_score = box_interference_score(chaser_pos, runner_pos, old_box_pos)
    old_box_blocks_path = box_blocks_between_agents(chaser_pos, runner_pos, old_box_pos)

    chaser_state = make_agent_state(chaser_pos, runner_pos, CHASER_SPEED)
    runner_state = make_agent_state(runner_pos, chaser_pos, RUNNER_SPEED)

    chaser_frozen = is_chaser_frozen(step_index)

    # 시작 후 CHASER_FREEZE_SECONDS 동안 술래는 움직일 수 없다.
    # 이 기간에는 술래의 행동 선택과 Q 업데이트를 하지 않는다.
    if chaser_frozen:
        chaser_action = None
        chaser_control = "FROZEN"
        new_chaser_pos = chaser_pos
        chaser_hit_wall = False
        chaser_pushed_box = False
        chaser_box_blocked = False
    else:
        chaser_action, chaser_control = choose_chaser_action(
            chaser_ai,
            chaser_state,
            chaser_pos,
            runner_pos,
            epsilon,
        )
        new_chaser_pos, chaser_hit_wall, chaser_pushed_box, chaser_box_blocked = move_position_with_box(
            chaser_pos,
            chaser_action,
            CHASER_SPEED,
            other_agent_pos=runner_pos,
        )

    runner_valid_actions = get_valid_actions_map_aware(
        runner_pos,
        RUNNER_SPEED,
        other_agent_pos=new_chaser_pos,
        allow_box_push=RUNNER_CAN_PUSH_BOX,
    )

    runner_action = choose_action_from_q_with_mask(
        runner_ai,
        runner_state,
        epsilon,
        runner_valid_actions,
    )

    new_runner_pos, runner_hit_wall, runner_pushed_box, runner_box_blocked = move_position_with_box(
        runner_pos,
        runner_action,
        RUNNER_SPEED,
        other_agent_pos=new_chaser_pos,
    )

    new_distance = distance(new_chaser_pos, new_runner_pos)
    chaser_reward = -0.01 + (old_distance - new_distance) * 0.02
    runner_reward = 0.01 + (new_distance - old_distance) * 0.02

    if chaser_hit_wall:
        chaser_reward -= WALL_HIT_PENALTY
        if distance(old_chaser_pos, new_chaser_pos) < 1e-6:
            chaser_reward -= STUCK_PENALTY
    if runner_hit_wall:
        runner_reward -= WALL_HIT_PENALTY
        if distance(old_runner_pos, new_runner_pos) < 1e-6:
            runner_reward -= STUCK_PENALTY

    new_box_interference_score = box_interference_score(new_chaser_pos, new_runner_pos, BOX_POS)
    new_box_blocks_path = box_blocks_between_agents(new_chaser_pos, new_runner_pos, BOX_POS)
    interference_progress = new_box_interference_score - old_box_interference_score

    if chaser_pushed_box:
        chaser_reward -= CHASER_BOX_PUSH_PENALTY

    if runner_pushed_box:
        runner_reward += BOX_PUSH_REWARD

        # 도망자가 상자를 밀어서 술래-도망자 사이의 경로를 더 잘 막으면 보상
        if interference_progress > 0:
            runner_reward += interference_progress * BOX_INTERFERENCE_PROGRESS_REWARD

        # 반대로 상자를 밀었는데 방해 효과가 줄어들면 약간 벌점
        elif interference_progress < 0:
            runner_reward += interference_progress * (BOX_INTERFERENCE_PROGRESS_REWARD * 0.35)

    if chaser_box_blocked:
        chaser_reward -= BOX_STUCK_PENALTY
    if runner_box_blocked:
        runner_reward -= BOX_STUCK_PENALTY

    # 상자가 실제로 술래와 도망자 사이를 막는 상태
    if new_box_blocks_path:
        runner_reward += BOX_INTERFERENCE_STEP_REWARD

        # 새로 막는 데 성공한 순간은 강하게 보상
        if not old_box_blocks_path:
            runner_reward += BOX_INTERFERENCE_REWARD
            chaser_reward -= 2.0

        # 술래가 가까울수록 상자로 방해하는 가치가 크다.
        if new_distance <= CAPTURE_RADIUS * 3.0:
            runner_reward += BOX_CLOSE_THREAT_BONUS

    winner = None
    result_text = "running"
    capture_blocked_by_obstacle = new_distance <= CAPTURE_RADIUS and is_line_of_sight_blocked(new_chaser_pos, new_runner_pos)

    if can_capture(new_chaser_pos, new_runner_pos):
        chaser_reward += 30.0
        runner_reward -= 30.0
        winner = "chaser"
        result_text = "chaser_win"
    elif capture_blocked_by_obstacle:
        chaser_reward -= 0.5

        # 벽보다 상자가 술래를 막았을 때 더 큰 보상
        if is_blocked_by_box(new_chaser_pos, new_runner_pos):
            runner_reward += CAPTURE_BLOCKED_BY_BOX_REWARD
            result_text = "capture_blocked_by_box"
        else:
            runner_reward += 0.5
            result_text = "capture_blocked_by_obstacle"
    elif step_index >= max_steps:
        chaser_reward -= 10.0
        runner_reward += 10.0
        winner = "runner"
        result_text = "runner_win"

    next_chaser_state = make_agent_state(new_chaser_pos, new_runner_pos, CHASER_SPEED)
    next_runner_state = make_agent_state(new_runner_pos, new_chaser_pos, RUNNER_SPEED)

    if training:
        if not chaser_frozen:
            chaser_ai.update_q(chaser_state, chaser_action, chaser_reward, next_chaser_state)
        runner_ai.update_q(runner_state, runner_action, runner_reward, next_runner_state)

    chaser_stuck_detected = is_chaser_strategically_stuck(new_chaser_pos, new_runner_pos)
    update_chaser_stuck_history(new_chaser_pos, new_runner_pos)

    info = {
        "step": step_index,
        "old_chaser_pos": old_chaser_pos,
        "old_runner_pos": old_runner_pos,
        "new_chaser_pos": new_chaser_pos,
        "new_runner_pos": new_runner_pos,
        "old_box_pos": old_box_pos,
        "new_box_pos": BOX_POS,
        "chaser_action": chaser_action,
        "runner_action": runner_action,
        "old_distance": old_distance,
        "new_distance": new_distance,
        "chaser_reward": chaser_reward,
        "runner_reward": runner_reward,
        "chaser_pushed_box": chaser_pushed_box,
        "runner_pushed_box": runner_pushed_box,
        "chaser_frozen": chaser_frozen,
        "chaser_control": chaser_control,
        "chaser_forced_route": chaser_control == "DOOR_ROUTE",
        "chaser_stuck_detected": chaser_stuck_detected,
        "runner_valid_action_count": len(runner_valid_actions),
        "capture_blocked_by_obstacle": capture_blocked_by_obstacle,
        "old_box_interference_score": old_box_interference_score,
        "new_box_interference_score": new_box_interference_score,
        "old_box_blocks_path": old_box_blocks_path,
        "new_box_blocks_path": new_box_blocks_path,
        "interference_progress": interference_progress,
        "winner": winner,
        "result": result_text,
    }
    return new_chaser_pos, new_runner_pos, winner, info


def play_one_episode(chaser_ai, runner_ai, epsilon, training=True, record=False):
    reset_box_position()
    reset_chaser_stuck_history()
    chaser_pos, runner_pos = get_start_positions()
    trajectory = []
    max_steps = get_max_steps()
    for step in range(1, max_steps + 1):
        chaser_pos, runner_pos, winner, info = simulate_one_step(
            chaser_ai, runner_ai, chaser_pos, runner_pos, epsilon, step, max_steps, training=training
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


# -----------------------------
# 학습 / 평가 / 시연
# -----------------------------
def train_episodes_fast(chaser_ai, runner_ai, episodes, total_episodes_done):
    chaser_wins = 0
    runner_wins = 0
    total_steps = 0
    progress_interval = max(1, episodes // 10)
    for i in range(1, episodes + 1):
        epsilon = get_epsilon(total_episodes_done + i)
        winner, steps, _ = play_one_episode(chaser_ai, runner_ai, epsilon, training=True, record=False)
        total_steps += steps
        if winner == "chaser":
            chaser_wins += 1
        else:
            runner_wins += 1
        if episodes >= 1000 and i % progress_interval == 0:
            print(f"fast 학습 진행: {i}/{episodes} ({i / episodes * 100:.0f}%) | 벽 캐시: {len(WALL_SENSOR_CACHE)}")

    print("\n===== 빠른 추가 학습 결과 =====")
    print(f"이번 실행에서 학습한 게임 수: {episodes}")
    print(f"이번 실행 술래 승: {chaser_wins}")
    print(f"이번 실행 도망자 승: {runner_wins}")
    print(f"이번 실행 평균 게임 길이: {total_steps * SIM_SECONDS_PER_STEP / episodes:.2f}초")
    print(f"현재 epsilon: {get_epsilon(total_episodes_done + episodes):.3f}")
    return total_episodes_done + episodes


def evaluate(chaser_ai, runner_ai, total_episodes_done, games=200):
    chaser_wins = 0
    runner_wins = 0
    total_steps = 0
    for _ in range(games):
        winner, steps, _ = play_one_episode(chaser_ai, runner_ai, epsilon=0.0, training=False, record=False)
        total_steps += steps
        if winner == "chaser":
            chaser_wins += 1
        else:
            runner_wins += 1
    print("\n===== 평가 결과 =====")
    print(f"지금까지 누적 학습한 게임 수: {total_episodes_done}")
    print(f"평가 게임 수: {games}")
    print(f"도망자가 버텨야 하는 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"술래 승률: {chaser_wins / games * 100:.1f}%")
    print(f"도망자 승률: {runner_wins / games * 100:.1f}%")
    print(f"평균 게임 길이: {total_steps * SIM_SECONDS_PER_STEP / games:.2f}초")


def format_pos(pos):
    return f"({pos[0]:6.1f}, {pos[1]:6.1f})"


def demo_text(chaser_ai, runner_ai):
    winner, steps, trajectory = play_one_episode(chaser_ai, runner_ai, epsilon=0.0, training=False, record=True)
    print("\n===== 방 + 상자 텍스트 시연 =====")
    print(f"경기장 크기: {ARENA_SIZE} x {ARENA_SIZE}")
    print(f"방 크기: {ROOM_SIZE} x {ROOM_SIZE}")
    print(f"방 위치: x={ROOM_LEFT:.1f}~{ROOM_RIGHT:.1f}, y={ROOM_BOTTOM:.1f}~{ROOM_TOP:.1f}")
    print(f"출입구: 좌우 벽, y={DOOR_BOTTOM:.1f}~{DOOR_TOP:.1f}, 높이={DOOR_SIZE}")
    print(f"상자 크기: {BOX_SIZE} x {BOX_SIZE}, 매 판 시작 위치: {BOX_START_POS}")
    print(f"술래 속도: {CHASER_SPEED}, 도망자 속도: {RUNNER_SPEED}")
    print(f"잡힘 반경: {CAPTURE_RADIUS}")
    print(f"도망자가 버텨야 하는 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"술래 시작 정지 시간: {CHASER_FREEZE_SECONDS:.1f}초")
    print(f"시작 위치 모드: {START_MODE}\n")

    for item in trajectory:
        elapsed_seconds = item["step"] * SIM_SECONDS_PER_STEP
        remain_seconds = max(0.0, get_survive_seconds() - elapsed_seconds)
        print(f"===== 시간 {elapsed_seconds:.0f}초 / 남은 시간 {remain_seconds:.0f}초 =====")
        print(f"술래   : {format_pos(item['old_chaser_pos'])} -> {format_pos(item['new_chaser_pos'])} / 행동: {action_label(item['chaser_action'])} / 제어: {item.get('chaser_control', 'Q')} / 헛돎감지: {item.get('chaser_stuck_detected', False)}")
        print(f"도망자 : {format_pos(item['old_runner_pos'])} -> {format_pos(item['new_runner_pos'])} / 행동: {action_label(item['runner_action'])}")
        print(f"상자   : {format_pos(item['old_box_pos'])} -> {format_pos(item['new_box_pos'])}")
        print(f"상자 방해 상태: {box_interference_status_text(item['new_chaser_pos'], item['new_runner_pos'], item['new_box_pos'])} / 방해 점수: {item.get('old_box_interference_score', 0):.2f} -> {item.get('new_box_interference_score', 0):.2f}")
        print(f"거리 변화: {item['old_distance']:.2f} -> {item['new_distance']:.2f}")
        print(f"보상: 술래 {item['chaser_reward']:.3f}, 도망자 {item['runner_reward']:.3f}")
        if item.get("chaser_frozen", False):
            print("상황 해석: 술래는 시작 대기 시간이라 움직이지 못했다.")
        if item['chaser_pushed_box']:
            print("상황 해석: 술래가 상자를 밀었다.")
        if item['runner_pushed_box']:
            print("상황 해석: 도망자가 상자를 밀었다.")
            if item.get("interference_progress", 0) > 0:
                print("상황 해석: 상자가 술래의 추격 경로를 더 잘 막는 위치로 이동했다.")
        if item.get("new_box_blocks_path", False):
            print("상황 해석: 상자가 술래와 도망자 사이를 가로막고 있다.")

        if item.get('capture_blocked_by_obstacle', False):
            print("상황 해석: 거리는 가깝지만 벽/상자가 사이를 막고 있어 잡지 못했다.")
        elif item['new_distance'] <= CAPTURE_RADIUS:
            print("상황 해석: 술래가 도망자를 잡았다.")
        elif item['new_distance'] < item['old_distance']:
            print("상황 해석: 술래가 도망자에게 가까워졌다.")
        elif item['new_distance'] > item['old_distance']:
            print("상황 해석: 도망자가 거리를 벌렸다.")
        else:
            print("상황 해석: 거리가 거의 변하지 않았다.")
        print()

    if winner == 'chaser':
        print(f"결과: 술래 승리! {steps * SIM_SECONDS_PER_STEP:.0f}초 만에 잡았다.")
    else:
        print(f"결과: 도망자 승리! {SURVIVE_MINUTES:.2f}분 동안 버텼다.")


def create_gui_window(title):
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

    def to_canvas(pos):
        x = padding + pos[0] * scale
        y = padding + (ARENA_SIZE - pos[1]) * scale
        return x, y

    for wall in WALLS:
        x1, y1 = to_canvas((wall["x1"], wall["y1"]))
        x2, y2 = to_canvas((wall["x2"], wall["y2"]))
        wall_width = max(1, int(wall["thickness"] * scale))
        fill = "black" if wall.get("kind") == "room" else "gray"
        canvas.create_line(x1, y1, x2, y2, width=wall_width, fill=fill)

    # 출입구 gap을 눈에 띄게 하기 위한 점선 표시
    for x in (ROOM_LEFT, ROOM_RIGHT):
        y1 = padding + (ARENA_SIZE - DOOR_BOTTOM) * scale
        y2 = padding + (ARENA_SIZE - DOOR_TOP) * scale
        cx = padding + x * scale
        canvas.create_line(cx, y1, cx, y2, width=2, fill="lightgray", dash=(3, 3))

    chaser_radius = 8
    runner_radius = 8
    chaser_obj = canvas.create_oval(0, 0, 0, 0, fill="red")
    runner_obj = canvas.create_oval(0, 0, 0, 0, fill="blue")
    capture_obj = canvas.create_oval(0, 0, 0, 0, outline="red", dash=(4, 4))
    box_obj = canvas.create_rectangle(0, 0, 0, 0, outline="#f0b400", width=3)

    def update_circle(obj, pos, radius):
        x, y = to_canvas(pos)
        canvas.coords(obj, x - radius, y - radius, x + radius, y + radius)

    def update_capture_circle(pos):
        x, y = to_canvas(pos)
        r = CAPTURE_RADIUS * scale
        canvas.coords(capture_obj, x - r, y - r, x + r, y + r)

    def update_box(pos):
        rect = box_rect(pos)
        x1, y1 = to_canvas((rect["left"], rect["top"]))
        x2, y2 = to_canvas((rect["right"], rect["bottom"]))
        canvas.coords(box_obj, x1, y1, x2, y2)

    return {
        "root": root,
        "canvas": canvas,
        "info_label": info_label,
        "chaser_obj": chaser_obj,
        "runner_obj": runner_obj,
        "capture_obj": capture_obj,
        "box_obj": box_obj,
        "chaser_radius": chaser_radius,
        "runner_radius": runner_radius,
        "update_circle": update_circle,
        "update_capture_circle": update_capture_circle,
        "update_box": update_box,
    }


def demo_gui(chaser_ai, runner_ai):
    try:
        import tkinter  # noqa
    except Exception:
        print("tkinter를 사용할 수 없어 GUI 시연을 실행할 수 없다.")
        return

    gui = create_gui_window("Continuous Tag AI Simulation - Room & Box Demo")
    winner, steps, trajectory = play_one_episode(chaser_ai, runner_ai, epsilon=0.0, training=False, record=True)
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
        box_pos = item["new_box_pos"]
        gui["update_circle"](gui["chaser_obj"], chaser_pos, gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], runner_pos, gui["runner_radius"])
        gui["update_capture_circle"](chaser_pos)
        gui["update_box"](box_pos)
        elapsed_seconds = item["step"] * SIM_SECONDS_PER_STEP
        remain_seconds = max(0.0, get_survive_seconds() - elapsed_seconds)
        info_label.config(
            text=(
                f"Demo | 시간: {elapsed_seconds:.0f}초 / 남은 시간: {remain_seconds:.0f}초\n"
                f"거리: {item['new_distance']:.2f} | 술래: {action_label(item['chaser_action'])}({item.get('chaser_control', 'Q')}) | 도망자: {action_label(item['runner_action'])}\n"
                f"상자 위치: ({box_pos[0]:.1f}, {box_pos[1]:.1f}) | 상자 방해: {box_interference_status_text(chaser_pos, runner_pos, box_pos)}\n"
                f"배속: {TRAIN_SPEED_MULTIPLIER:.1f}x | 술래 대기: {CHASER_FREEZE_SECONDS:.0f}초"
            )
        )
        root.after(get_animation_delay_ms(), lambda: animate(index + 1))

    if trajectory:
        gui["update_circle"](gui["chaser_obj"], trajectory[0]["old_chaser_pos"], gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], trajectory[0]["old_runner_pos"], gui["runner_radius"])
        gui["update_capture_circle"](trajectory[0]["old_chaser_pos"])
        gui["update_box"](trajectory[0]["old_box_pos"])

    animate()
    root.mainloop()


def train_episodes_visual(chaser_ai, runner_ai, episodes, total_episodes_done):
    try:
        import tkinter  # noqa
    except Exception:
        print("tkinter를 사용할 수 없어 GUI 학습을 실행할 수 없다.")
        print("대신 빠른 학습으로 진행한다.")
        return train_episodes_fast(chaser_ai, runner_ai, episodes, total_episodes_done)

    gui = create_gui_window("Continuous Tag AI Simulation - Room & Box Visual Training")
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
        reset_box_position()
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
        state["epsilon"] = get_epsilon(state["total_episodes_done"] + 1)
        gui["update_circle"](gui["chaser_obj"], state["chaser_pos"], gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], state["runner_pos"], gui["runner_radius"])
        gui["update_capture_circle"](state["chaser_pos"])
        gui["update_box"](BOX_POS)
        root.after(300, animate_step)

    def animate_step():
        max_steps = get_max_steps()
        state["current_step"] += 1
        chaser_pos, runner_pos, winner, info = simulate_one_step(
            chaser_ai, runner_ai,
            state["chaser_pos"], state["runner_pos"],
            state["epsilon"], state["current_step"], max_steps,
            training=True,
        )
        state["chaser_pos"] = chaser_pos
        state["runner_pos"] = runner_pos
        gui["update_circle"](gui["chaser_obj"], chaser_pos, gui["chaser_radius"])
        gui["update_circle"](gui["runner_obj"], runner_pos, gui["runner_radius"])
        gui["update_capture_circle"](chaser_pos)
        gui["update_box"](BOX_POS)
        elapsed_seconds = state["current_step"] * SIM_SECONDS_PER_STEP
        remain_seconds = max(0.0, get_survive_seconds() - elapsed_seconds)
        info_label.config(
            text=(
                f"방+상자 시각화 학습 중\n"
                f"현재 판: {state['completed'] + 1} / {episodes} | 누적 학습 게임 수: {state['total_episodes_done']}\n"
                f"시간: {elapsed_seconds:.0f}초 / 남은 시간: {remain_seconds:.0f}초 | 생존 목표: {SURVIVE_MINUTES:.2f}분\n"
                f"거리: {info['new_distance']:.2f} | 술래 행동: {action_label(info['chaser_action'])}({info.get('chaser_control', 'Q')}) | 도망자 행동: {action_label(info['runner_action'])}\n"
                f"상자: ({BOX_POS[0]:.1f}, {BOX_POS[1]:.1f}) | 상자 방해: {box_interference_status_text(chaser_pos, runner_pos, BOX_POS)}\n"
                f"epsilon: {state['epsilon']:.3f} | 배속: {TRAIN_SPEED_MULTIPLIER:.1f}x | 술래 대기: {CHASER_FREEZE_SECONDS:.0f}초\n"
                f"술래 승: {state['chaser_wins']} | 도망자 승: {state['runner_wins']}"
            )
        )

        if winner is not None:
            state["completed"] += 1
            state["total_episodes_done"] += 1
            state["total_steps"] += state["current_step"]
            if winner == "chaser":
                state["chaser_wins"] += 1
            else:
                state["runner_wins"] += 1
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


# -----------------------------
# 저장 / 불러오기 / 초기화
# -----------------------------
def save_learning_data(chaser_ai, runner_ai, total_episodes_done):
    data = {
        "total_episodes_done": total_episodes_done,
        "chaser_score": chaser_ai.score,
        "runner_score": runner_ai.score,
        "chaser_q_table": dict(chaser_ai.q_table),
        "runner_q_table": dict(runner_ai.q_table),
        "settings": {
            "survive_minutes": SURVIVE_MINUTES,
            "chaser_freeze_seconds": CHASER_FREEZE_SECONDS,
            "train_speed_multiplier": TRAIN_SPEED_MULTIPLIER,
            "start_mode": START_MODE,
            "custom_walls": [wall for wall in WALLS if wall.get("kind") == "custom"],
            "box_pos": BOX_POS,
            "box_start_pos": BOX_START_POS,
            "wall_sensor_cache_grid": WALL_SENSOR_CACHE_GRID,
            "chaser_pathfinding_enabled": CHASER_PATHFINDING_ENABLED,
            "chaser_pathfinding_rate": CHASER_PATHFINDING_RATE,
            "path_grid_size": PATH_GRID_SIZE,
            "chaser_path_mode": CHASER_PATH_MODE,
            "chaser_astar_fallback_enabled": CHASER_ASTAR_FALLBACK_ENABLED,
            "chaser_astar_fallback_rate": CHASER_ASTAR_FALLBACK_RATE,
            "chaser_stuck_detection_enabled": CHASER_STUCK_DETECTION_ENABLED,
            "chaser_stuck_window_steps": CHASER_STUCK_WINDOW_STEPS,
            "chaser_stuck_min_progress": CHASER_STUCK_MIN_PROGRESS,
            "map_aware_action_masking": MAP_AWARE_ACTION_MASKING,
            "map_aware_exploration": MAP_AWARE_EXPLORATION,
            "chaser_can_push_box": CHASER_CAN_PUSH_BOX,
            "runner_can_push_box": RUNNER_CAN_PUSH_BOX,
            "wall_clearance": WALL_CLEARANCE,
            "chaser_door_waypoint_enabled": CHASER_DOOR_WAYPOINT_ENABLED,
            "chaser_force_door_route": CHASER_FORCE_DOOR_ROUTE,
            "door_waypoint_offset": DOOR_WAYPOINT_OFFSET,
            "box_interference_progress_reward": BOX_INTERFERENCE_PROGRESS_REWARD,
            "box_interference_reward": BOX_INTERFERENCE_REWARD,
            "box_interference_step_reward": BOX_INTERFERENCE_STEP_REWARD,
        },
    }
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"\n학습 데이터 저장 완료: {SAVE_FILE}")


def load_learning_data(chaser_ai, runner_ai):
    global SURVIVE_MINUTES, CHASER_FREEZE_SECONDS, TRAIN_SPEED_MULTIPLIER, START_MODE, WALL_SENSOR_CACHE_GRID, BOX_POS, BOX_START_POS
    global CHASER_PATHFINDING_ENABLED, CHASER_PATHFINDING_RATE, PATH_GRID_SIZE
    global CHASER_PATH_MODE, CHASER_ASTAR_FALLBACK_ENABLED, CHASER_ASTAR_FALLBACK_RATE
    global CHASER_STUCK_DETECTION_ENABLED, CHASER_STUCK_WINDOW_STEPS, CHASER_STUCK_MIN_PROGRESS
    global MAP_AWARE_ACTION_MASKING, MAP_AWARE_EXPLORATION, CHASER_CAN_PUSH_BOX, RUNNER_CAN_PUSH_BOX
    global CHASER_DOOR_WAYPOINT_ENABLED
    global CHASER_FORCE_DOOR_ROUTE, DOOR_WAYPOINT_OFFSET
    rebuild_room_walls()
    reset_box_position()
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
    CHASER_FREEZE_SECONDS = float(settings.get("chaser_freeze_seconds", CHASER_FREEZE_SECONDS))
    TRAIN_SPEED_MULTIPLIER = float(settings.get("train_speed_multiplier", TRAIN_SPEED_MULTIPLIER))
    START_MODE = settings.get("start_mode", START_MODE)
    if START_MODE not in ("random", "fixed"):
        START_MODE = "random"
    WALL_SENSOR_CACHE_GRID = float(settings.get("wall_sensor_cache_grid", WALL_SENSOR_CACHE_GRID))
    CHASER_PATHFINDING_ENABLED = bool(settings.get("chaser_pathfinding_enabled", CHASER_PATHFINDING_ENABLED))
    CHASER_PATHFINDING_RATE = float(settings.get("chaser_pathfinding_rate", CHASER_PATHFINDING_RATE))
    PATH_GRID_SIZE = float(settings.get("path_grid_size", PATH_GRID_SIZE))
    CHASER_PATH_MODE = settings.get("chaser_path_mode", CHASER_PATH_MODE)
    if CHASER_PATH_MODE not in ("fast", "astar"):
        CHASER_PATH_MODE = "fast"
    CHASER_ASTAR_FALLBACK_ENABLED = bool(settings.get("chaser_astar_fallback_enabled", CHASER_ASTAR_FALLBACK_ENABLED))
    CHASER_ASTAR_FALLBACK_RATE = float(settings.get("chaser_astar_fallback_rate", CHASER_ASTAR_FALLBACK_RATE))
    CHASER_STUCK_DETECTION_ENABLED = bool(settings.get("chaser_stuck_detection_enabled", CHASER_STUCK_DETECTION_ENABLED))
    CHASER_STUCK_WINDOW_STEPS = int(settings.get("chaser_stuck_window_steps", CHASER_STUCK_WINDOW_STEPS))
    CHASER_STUCK_MIN_PROGRESS = float(settings.get("chaser_stuck_min_progress", CHASER_STUCK_MIN_PROGRESS))
    MAP_AWARE_ACTION_MASKING = bool(settings.get("map_aware_action_masking", MAP_AWARE_ACTION_MASKING))
    MAP_AWARE_EXPLORATION = bool(settings.get("map_aware_exploration", MAP_AWARE_EXPLORATION))
    CHASER_CAN_PUSH_BOX = bool(settings.get("chaser_can_push_box", CHASER_CAN_PUSH_BOX))
    RUNNER_CAN_PUSH_BOX = bool(settings.get("runner_can_push_box", RUNNER_CAN_PUSH_BOX))
    CHASER_DOOR_WAYPOINT_ENABLED = bool(settings.get("chaser_door_waypoint_enabled", CHASER_DOOR_WAYPOINT_ENABLED))
    CHASER_FORCE_DOOR_ROUTE = bool(settings.get("chaser_force_door_route", CHASER_FORCE_DOOR_ROUTE))
    DOOR_WAYPOINT_OFFSET = float(settings.get("door_waypoint_offset", DOOR_WAYPOINT_OFFSET))

    rebuild_room_walls()
    for wall in settings.get("custom_walls", []):
        try:
            WALLS.append(clamp_wall_to_arena(make_wall(
                wall["x1"], wall["y1"], wall["x2"], wall["y2"], wall.get("thickness", DEFAULT_WALL_THICKNESS), kind="custom"
            )))
        except Exception:
            pass

    candidate_box_start_pos = tuple(settings.get("box_start_pos", BOX_START_POS))
    if can_place_box(candidate_box_start_pos):
        BOX_START_POS = candidate_box_start_pos

    candidate_box_pos = tuple(settings.get("box_pos", BOX_START_POS))
    if can_place_box(candidate_box_pos):
        BOX_POS = candidate_box_pos
    else:
        BOX_POS = BOX_START_POS
    clear_wall_sensor_cache()

    total_episodes_done = data.get("total_episodes_done", 0)
    print("저장된 학습 데이터를 불러왔다.")
    print(f"누적 학습 게임 수: {total_episodes_done}")
    print(f"누적 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")
    print(f"도망자 생존 목표 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"술래 시작 정지 시간: {CHASER_FREEZE_SECONDS:.1f}초")
    print(f"현재 시각화 배속: {TRAIN_SPEED_MULTIPLIER:.1f}x")
    print(f"현재 시작 위치 모드: {START_MODE}")
    print(f"방 벽 포함 총 벽 개수: {len(WALLS)}")
    print(f"상자 위치: {BOX_POS}, 상자 크기: {BOX_SIZE}")
    return total_episodes_done


def reset_learning_data(chaser_ai, runner_ai):
    chaser_ai.clear_learning_data()
    runner_ai.clear_learning_data()
    rebuild_room_walls()
    reset_box_position()
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)
    print("\n전체 학습 데이터를 초기화했다.")
    print("Q-table, 점수, 저장 파일을 삭제했다.")
    print("방 구조와 상자는 기본값으로 유지된다.")
    return 0


# -----------------------------
# 명령어 처리
# -----------------------------
def command_to_episodes(command):
    if command == "":
        return 1
    preset = {"1": 1, "train1": 1, "10": 10, "train10": 10, "100": 100, "train100": 100, "1000": 1000, "train1000": 1000}
    if command in preset:
        return preset[command]
    if command.isdigit():
        value = int(command)
        if value > 0:
            return value
    return None


def command_to_fast_episodes(command):
    parts = command.split()
    if len(parts) != 2:
        return None
    if parts[0] not in ("fast", "f"):
        return None
    if not parts[1].isdigit():
        return None
    value = int(parts[1])
    return value if value > 0 else None


def command_to_turbo_episodes(command):
    """
    turbo 숫자:
    wallcache를 크게 잡고 빠르게 학습한다.

    예:
    turbo 10000
    """
    parts = command.split()
    if len(parts) != 2:
        return None
    if parts[0] not in ("turbo", "t"):
        return None
    if not parts[1].isdigit():
        return None
    value = int(parts[1])
    return value if value > 0 else None


def handle_speed_command(command):
    global TRAIN_SPEED_MULTIPLIER
    parts = command.split()
    if len(parts) != 2 or parts[0] not in ("speed", "배속"):
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
    global SURVIVE_MINUTES
    parts = command.split()
    if len(parts) != 2 or parts[0] not in ("minutes", "minute", "time", "m", "분"):
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


def handle_freeze_command(command):
    """
    술래 시작 정지 시간 변경 명령어.

    예:
    freeze 10
    freeze 0
    freeze 5.5
    """
    global CHASER_FREEZE_SECONDS

    parts = command.split()

    if len(parts) != 2:
        return False

    if parts[0] not in ("freeze", "wait", "정지"):
        return False

    try:
        value = float(parts[1])
    except ValueError:
        print("정지 시간은 숫자로 입력해야 한다. 예: freeze 10")
        return True

    if value < 0:
        print("정지 시간은 0초 이상이어야 한다.")
        return True

    CHASER_FREEZE_SECONDS = value
    print(f"술래 시작 정지 시간을 {CHASER_FREEZE_SECONDS:.1f}초로 변경했다.")
    return True


def handle_start_command(command):
    global START_MODE
    parts = command.split()
    if len(parts) != 2 or parts[0] not in ("start", "시작"):
        return False
    mode = parts[1]
    if mode == "랜덤":
        mode = "random"
    elif mode == "고정":
        mode = "fixed"
    if mode not in ("random", "fixed"):
        print("시작 위치 모드는 random 또는 fixed만 가능하다.")
        return True
    START_MODE = mode
    if START_MODE == "random":
        print("시작 위치 모드를 random으로 변경했다.")
    else:
        print("시작 위치 모드를 fixed로 변경했다.")
        print(f"술래 고정 시작 위치: {FIXED_CHASER_POS}")
        print(f"도망자 고정 시작 위치: {FIXED_RUNNER_POS}")
    return True


def handle_wall_command(command):
    parts = command.split()
    if not parts or parts[0] not in ("wall", "벽"):
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
            x1 = float(parts[2]); y1 = float(parts[3]); x2 = float(parts[4]); y2 = float(parts[5])
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
        clear_custom_walls()
        return True
    if action in ("remove", "delete", "삭제"):
        if len(parts) != 3 or not parts[2].isdigit():
            print("삭제할 벽 번호를 입력해야 한다. 예: wall remove 1")
            return True
        remove_wall(int(parts[2]))
        return True
    print("알 수 없는 wall 명령어다.")
    return True


def handle_wall_cache_command(command):
    global WALL_SENSOR_CACHE_GRID
    parts = command.split()
    if len(parts) != 2 or parts[0] not in ("wallcache", "cachegrid", "벽캐시"):
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
    return True


def handle_box_command(command):
    global BOX_POS
    parts = command.split()
    if not parts or parts[0] not in ("box", "상자"):
        return False
    if len(parts) == 1:
        print("상자 명령어:")
        print("box reset")
        print("box pos")
        print("box set x y       # 현재 상자 위치 변경")
        print("boxstart x y      # 매 판 시작 상자 위치 변경")
        return True
    action = parts[1]
    if action in ("reset", "초기화"):
        reset_box_position()
        print(f"상자 위치를 기본값 {BOX_POS}로 되돌렸다.")
        return True
    if action in ("pos", "position", "위치"):
        print(f"현재 상자 위치: {BOX_POS}")
        print(f"상자 크기: {BOX_SIZE} x {BOX_SIZE}")
        return True
    if action in ("set", "설정"):
        if len(parts) != 4:
            print("상자 위치 설정 형식: box set x y")
            return True
        try:
            x = float(parts[2]); y = float(parts[3])
        except ValueError:
            print("상자 좌표는 숫자로 입력해야 한다.")
            return True
        candidate = (x, y)
        if not can_place_box(candidate):
            print("해당 위치에는 상자를 놓을 수 없다. 벽이나 경기장 밖과 겹친다.")
            return True
        BOX_POS = candidate
        clear_wall_sensor_cache()
        print(f"상자 위치를 {BOX_POS}로 변경했다.")
        return True
    print("알 수 없는 box 명령어다.")
    return True


def handle_box_start_command(command):
    """
    매 판 시작 시 상자의 기본 위치를 변경한다.

    예:
    boxstart 250 390
    boxstart 180 390
    """
    global BOX_START_POS
    global BOX_POS

    parts = command.split()

    if len(parts) != 3:
        return False

    if parts[0] not in ("boxstart", "상자시작"):
        return False

    try:
        x = float(parts[1])
        y = float(parts[2])
    except ValueError:
        print("상자 시작 좌표는 숫자로 입력해야 한다. 예: boxstart 250 390")
        return True

    candidate = (x, y)

    if not can_place_box(candidate):
        print("해당 위치에는 상자를 시작시킬 수 없다.")
        print("상자가 벽이나 경기장 밖과 겹치지 않는 위치를 입력해야 한다.")
        return True

    BOX_START_POS = candidate
    BOX_POS = candidate
    clear_wall_sensor_cache()

    print(f"상자의 매 판 시작 위치를 {BOX_START_POS}로 변경했다.")
    print("이제 새 판이 시작될 때마다 상자가 이 위치에서 시작한다.")
    return True


def handle_path_command(command):
    """
    술래 길찾기 설정 명령어.

    path on
    path off
    pathrate 0.85
    pathgrid 25
    pathmode fast
    pathmode astar
    astarfallback 0.03
    stuck on
    stuck off
    stuckwindow 8
    stuckprogress 8
    """
    global CHASER_PATHFINDING_ENABLED
    global CHASER_PATHFINDING_RATE
    global PATH_GRID_SIZE
    global CHASER_PATH_MODE
    global CHASER_ASTAR_FALLBACK_RATE
    global CHASER_STUCK_DETECTION_ENABLED
    global CHASER_STUCK_WINDOW_STEPS
    global CHASER_STUCK_MIN_PROGRESS

    parts = command.split()

    if not parts:
        return False

    if parts[0] == "path":
        if len(parts) != 2:
            print("사용법: path on 또는 path off")
            return True

        if parts[1] in ("on", "켜기"):
            CHASER_PATHFINDING_ENABLED = True
            PATHFINDING_CACHE.clear()
            print("술래 A* 길찾기를 켰다.")
            return True

        if parts[1] in ("off", "끄기"):
            CHASER_PATHFINDING_ENABLED = False
            PATHFINDING_CACHE.clear()
            print("술래 A* 길찾기를 껐다.")
            return True

        print("사용법: path on 또는 path off")
        return True

    if parts[0] == "pathrate":
        if len(parts) != 2:
            print("사용법: pathrate 0.85")
            return True

        try:
            value = float(parts[1])
        except ValueError:
            print("pathrate는 0~1 사이 숫자로 입력해야 한다.")
            return True

        CHASER_PATHFINDING_RATE = clamp(value, 0.0, 1.0)
        print(f"술래 길찾기 사용 비율을 {CHASER_PATHFINDING_RATE:.2f}로 변경했다.")
        return True

    if parts[0] == "pathgrid":
        if len(parts) != 2:
            print("사용법: pathgrid 25")
            return True

        try:
            value = float(parts[1])
        except ValueError:
            print("pathgrid는 숫자로 입력해야 한다.")
            return True

        if value < 10:
            print("pathgrid가 너무 작으면 느려질 수 있다. 10 이상을 권장한다.")
            return True

        PATH_GRID_SIZE = value
        PATHFINDING_CACHE.clear()
        print(f"술래 A* 길찾기 격자를 {PATH_GRID_SIZE:.1f}로 변경했다.")
        print("값이 작을수록 정밀하지만 느리고, 값이 클수록 빠르지만 거칠어진다.")
        return True

    if parts[0] == "pathmode":
        if len(parts) != 2:
            print("사용법: pathmode fast 또는 pathmode astar")
            return True

        if parts[1] not in ("fast", "astar"):
            print("pathmode는 fast 또는 astar만 가능하다.")
            return True

        CHASER_PATH_MODE = parts[1]
        PATHFINDING_CACHE.clear()

        if CHASER_PATH_MODE == "fast":
            print("술래 길찾기 모드를 fast로 변경했다.")
            print("A*를 거의 쓰지 않고 8방향 빠른 장애물 회피 추격을 사용한다.")
        else:
            print("술래 길찾기 모드를 astar로 변경했다.")
            print("더 정밀하지만 학습 속도는 느려질 수 있다.")
        return True

    if parts[0] == "astarfallback":
        if len(parts) != 2:
            print("사용법: astarfallback 0.03")
            return True

        try:
            value = float(parts[1])
        except ValueError:
            print("astarfallback은 0~1 사이 숫자로 입력해야 한다.")
            return True

        CHASER_ASTAR_FALLBACK_RATE = clamp(value, 0.0, 1.0)
        print(f"A* fallback 확률을 {CHASER_ASTAR_FALLBACK_RATE:.3f}로 변경했다.")
        return True

    if parts[0] == "stuck":
        if len(parts) != 2:
            print("사용법: stuck on 또는 stuck off")
            return True

        if parts[1] in ("on", "켜기"):
            CHASER_STUCK_DETECTION_ENABLED = True
            print("술래 제자리걸음/헛돎 감지를 켰다.")
            return True

        if parts[1] in ("off", "끄기"):
            CHASER_STUCK_DETECTION_ENABLED = False
            print("술래 제자리걸음/헛돎 감지를 껐다.")
            return True

        print("사용법: stuck on 또는 stuck off")
        return True

    if parts[0] == "stuckwindow":
        if len(parts) != 2 or not parts[1].isdigit():
            print("사용법: stuckwindow 8")
            return True

        value = int(parts[1])
        if value < 3:
            print("stuckwindow는 3 이상이어야 한다.")
            return True

        CHASER_STUCK_WINDOW_STEPS = value
        print(f"술래 헛돎 감지 window를 {CHASER_STUCK_WINDOW_STEPS} step으로 변경했다.")
        return True

    if parts[0] == "stuckprogress":
        if len(parts) != 2:
            print("사용법: stuckprogress 8")
            return True

        try:
            value = float(parts[1])
        except ValueError:
            print("stuckprogress는 숫자로 입력해야 한다.")
            return True

        if value < 0:
            print("stuckprogress는 0 이상이어야 한다.")
            return True

        CHASER_STUCK_MIN_PROGRESS = value
        print(f"술래 헛돎 감지 최소 거리 감소량을 {CHASER_STUCK_MIN_PROGRESS:.1f}로 변경했다.")
        return True

    return False


def handle_map_command(command):
    """
    AI 지도 사전 인식 설정.

    map on
    map off
    mapexplore on/off
    chaserpush on/off
    runnerpush on/off
    """
    global MAP_AWARE_ACTION_MASKING
    global MAP_AWARE_EXPLORATION
    global CHASER_CAN_PUSH_BOX
    global RUNNER_CAN_PUSH_BOX
    global CHASER_DOOR_WAYPOINT_ENABLED
    global CHASER_FORCE_DOOR_ROUTE

    parts = command.split()

    if not parts:
        return False

    if parts[0] == "map":
        if len(parts) != 2:
            print("사용법: map on 또는 map off")
            return True

        if parts[1] in ("on", "켜기"):
            MAP_AWARE_ACTION_MASKING = True
            print("AI 지도 사전 인식을 켰다. 막힌 방향은 행동 선택지에서 제외된다.")
            return True

        if parts[1] in ("off", "끄기"):
            MAP_AWARE_ACTION_MASKING = False
            print("AI 지도 사전 인식을 껐다.")
            return True

        print("사용법: map on 또는 map off")
        return True

    if parts[0] == "mapexplore":
        if len(parts) != 2:
            print("사용법: mapexplore on 또는 mapexplore off")
            return True

        if parts[1] in ("on", "켜기"):
            MAP_AWARE_EXPLORATION = True
            print("탐험 중에도 막힌 방향을 제외한다.")
            return True

        if parts[1] in ("off", "끄기"):
            MAP_AWARE_EXPLORATION = False
            print("탐험 중 막힌 방향 제외를 껐다.")
            return True

        print("사용법: mapexplore on 또는 mapexplore off")
        return True

    if parts[0] == "chaserpush":
        if len(parts) != 2:
            print("사용법: chaserpush on 또는 chaserpush off")
            return True

        CHASER_CAN_PUSH_BOX = parts[1] in ("on", "켜기")
        print(f"술래 상자 밀기: {'ON' if CHASER_CAN_PUSH_BOX else 'OFF'}")
        return True

    if parts[0] == "forceroute":
        if len(parts) != 2:
            print("사용법: forceroute on 또는 forceroute off")
            return True

        if parts[1] in ("on", "켜기"):
            CHASER_FORCE_DOOR_ROUTE = True
            print("술래 강제 출입구 우회 경로를 켰다.")
            return True

        if parts[1] in ("off", "끄기"):
            CHASER_FORCE_DOOR_ROUTE = False
            print("술래 강제 출입구 우회 경로를 껐다.")
            return True

        print("사용법: forceroute on 또는 forceroute off")
        return True

    if parts[0] == "doorwaypoint":
        if len(parts) != 2:
            print("사용법: doorwaypoint on 또는 doorwaypoint off")
            return True

        if parts[1] in ("on", "켜기"):
            CHASER_DOOR_WAYPOINT_ENABLED = True
            print("술래 문 우회 waypoint를 켰다.")
            return True

        if parts[1] in ("off", "끄기"):
            CHASER_DOOR_WAYPOINT_ENABLED = False
            print("술래 문 우회 waypoint를 껐다.")
            return True

        print("사용법: doorwaypoint on 또는 doorwaypoint off")
        return True

    if parts[0] == "runnerpush":
        if len(parts) != 2:
            print("사용법: runnerpush on 또는 runnerpush off")
            return True

        RUNNER_CAN_PUSH_BOX = parts[1] in ("on", "켜기")
        print(f"도망자 상자 밀기: {'ON' if RUNNER_CAN_PUSH_BOX else 'OFF'}")
        return True

    return False


def print_menu():
    print("\n===== 명령어 =====")
    print("Enter 또는 1        : 1판 시각화 학습 + 저장")
    print("10 또는 train10     : 10판 시각화 학습 + 저장")
    print("100 또는 train100   : 100판 시각화 학습 + 저장")
    print("1000 또는 train1000 : 1000판 시각화 학습 + 저장")
    print("숫자 입력           : 해당 숫자만큼 시각화 학습 + 저장")
    print("fast 숫자           : GUI 없이 빠르게 추가 학습. 예: fast 1000")
    print("turbo 숫자          : wallcache를 크게 잡고 더 빠르게 학습. 예: turbo 10000")
    print("demo                : 학습 없이 현재 AI로 텍스트 시연")
    print("visual              : 학습 없이 현재 AI로 GUI 시연")
    print("eval                : 학습 없이 현재 AI 승률 평가")
    print("speed 숫자          : 학습/시연 화면 배속 변경. 예: speed 50")
    print("minutes 숫자        : 도망자 생존 목표 시간 변경. 예: minutes 0.5")
    print("freeze 숫자         : 매 판 시작 후 술래 정지 시간 설정. 예: freeze 10")
    print("start random        : 매 판 랜덤 위치에서 시작")
    print("start fixed         : 매 판 고정 위치에서 시작")
    print("wall add x1 y1 x2 y2 [두께] : 사용자 벽 추가")
    print("wall list           : 현재 벽 목록 보기")
    print("wall remove 번호    : 사용자 벽 삭제")
    print("wall clear          : 사용자 추가 벽 전체 삭제, 방 벽은 유지")
    print("box reset           : 상자 위치 초기화")
    print("box pos             : 상자 위치 확인")
    print("box set x y         : 현재 상자 위치 직접 설정")
    print("boxstart x y        : 매 판 시작 상자 위치 설정. 예: boxstart 250 390")
    print("wallcache 숫자      : 벽 감지 캐시 크기 설정. 예: wallcache 25")
    print("path on/off         : 술래 길찾기 켜기/끄기")
    print("pathmode fast/astar : 술래 길찾기 모드 설정. fast가 훨씬 빠름")
    print("pathrate 숫자       : 술래 길찾기 사용 비율 설정. 예: pathrate 0.9")
    print("pathgrid 숫자       : A* 격자 크기 설정. astar 모드에서 중요")
    print("astarfallback 숫자  : 헛돎 감지 시 A* 사용 확률. 예: astarfallback 0.2")
    print("stuck on/off        : 술래 제자리걸음/헛돎 감지 켜기/끄기")
    print("stuckwindow 숫자    : 헛돎 판단 step 수. 예: stuckwindow 8")
    print("stuckprogress 숫자  : 헛돎 판단 최소 거리 감소량. 예: stuckprogress 8")
    print("map on/off          : AI에게 맵 구조 사전 인식 제공")
    print("mapexplore on/off   : 탐험 중에도 막힌 방향 제외")
    print("chaserpush on/off   : 술래 상자 밀기 허용/금지")
    print("runnerpush on/off   : 도망자 상자 밀기 허용/금지")
    print("doorwaypoint on/off : 술래가 방 문으로 우회하는 waypoint 사용")
    print("forceroute on/off   : 방 안 도망자 추격 시 출입구 우회 경로 강제")
    print("reset               : 저장된 학습 데이터 전체 초기화")
    print("quit                : 저장 후 종료")
    print("\n----- 현재 설정 -----")
    print(f"도망자 생존 목표 시간: {SURVIVE_MINUTES:.2f}분")
    print(f"술래 시작 정지 시간: {CHASER_FREEZE_SECONDS:.1f}초")
    print(f"시각화 배속: {TRAIN_SPEED_MULTIPLIER:.1f}x")
    print(f"시작 위치 모드: {START_MODE}")
    print(f"방 위치: x={ROOM_LEFT:.1f}~{ROOM_RIGHT:.1f}, y={ROOM_BOTTOM:.1f}~{ROOM_TOP:.1f}")
    print(f"출입구: 좌우 벽, y={DOOR_BOTTOM:.1f}~{DOOR_TOP:.1f}, 높이={DOOR_SIZE}")
    print(f"상자 현재 위치: ({BOX_POS[0]:.1f}, {BOX_POS[1]:.1f}), 크기={BOX_SIZE}")
    print(f"상자 시작 위치: ({BOX_START_POS[0]:.1f}, {BOX_START_POS[1]:.1f})")
    print(f"AI-상자 충돌 반경: {AGENT_COLLISION_RADIUS}")
    print(f"상자 방해 상태: 아직 게임 중 좌표 기준으로 계산됨")
    print(f"상자 방해 보상: 순간 +{BOX_INTERFERENCE_REWARD}, 유지 +{BOX_INTERFERENCE_STEP_REWARD}/step")
    print(f"상자-추격경로 접근 보상 계수: {BOX_INTERFERENCE_PROGRESS_REWARD}")
    print(f"방 벽 포함 총 벽 개수: {len(WALLS)}")
    print(f"벽 감지 캐시 격자: {WALL_SENSOR_CACHE_GRID}")
    print(f"벽 감지 캐시 저장 수: {len(WALL_SENSOR_CACHE)}")
    print(f"술래 길찾기: {'ON' if CHASER_PATHFINDING_ENABLED else 'OFF'}")
    print(f"술래 길찾기 모드: {CHASER_PATH_MODE}")
    print(f"술래 길찾기 사용 비율: {CHASER_PATHFINDING_RATE:.2f}")
    print(f"헛돎 감지 시 A* 사용 확률: {CHASER_ASTAR_FALLBACK_RATE:.3f}")
    print(f"술래 헛돎 감지: {'ON' if CHASER_STUCK_DETECTION_ENABLED else 'OFF'}")
    print(f"헛돎 판단 window: {CHASER_STUCK_WINDOW_STEPS} step")
    print(f"헛돎 판단 최소 거리 감소량: {CHASER_STUCK_MIN_PROGRESS}")
    print(f"AI 지도 사전 인식: {'ON' if MAP_AWARE_ACTION_MASKING else 'OFF'}")
    print(f"탐험 중 막힌 방향 제외: {'ON' if MAP_AWARE_EXPLORATION else 'OFF'}")
    print(f"술래 상자 밀기: {'ON' if CHASER_CAN_PUSH_BOX else 'OFF'}")
    print(f"도망자 상자 밀기: {'ON' if RUNNER_CAN_PUSH_BOX else 'OFF'}")
    print(f"AI-벽 여유 거리: {WALL_CLEARANCE}")
    print(f"술래 문 우회 waypoint: {'ON' if CHASER_DOOR_WAYPOINT_ENABLED else 'OFF'}")
    print(f"술래 강제 출입구 우회: {'ON' if CHASER_FORCE_DOOR_ROUTE else 'OFF'}")
    print(f"술래 A* 격자: {PATH_GRID_SIZE}")
    print(f"술래 길찾기 캐시 저장 수: {len(PATHFINDING_CACHE)}")
    print(f"잡힘 반경: {CAPTURE_RADIUS}")
    print(f"술래 속도: {CHASER_SPEED}, 도망자 속도: {RUNNER_SPEED}")


# -----------------------------
# 메인 루프
# -----------------------------
def main():
    random.seed()
    rebuild_room_walls()
    reset_box_position()
    chaser_ai = QLearningAgent("술래 AI")
    runner_ai = QLearningAgent("도망자 AI")
    total_episodes_done = load_learning_data(chaser_ai, runner_ai)

    while True:
        print_menu()
        command = input("\n명령어 입력: ").strip().lower()

        if command.startswith("freeze ") or command.startswith("wait ") or command.startswith("정지 "):
            if handle_freeze_command(command):
                save_learning_data(chaser_ai, runner_ai, total_episodes_done)
                continue

        if handle_speed_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_minutes_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_freeze_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_start_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_wall_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_box_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_box_start_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_wall_cache_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_path_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue
        if handle_map_command(command):
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        turbo_episodes = command_to_turbo_episodes(command)
        if turbo_episodes is not None:
            if WALL_SENSOR_CACHE_GRID < 40:
                globals()["WALL_SENSOR_CACHE_GRID"] = 40.0
                clear_wall_sensor_cache()
                print("turbo 모드: wallcache를 40으로 올렸다.")
                print("속도는 빨라지지만 벽/상자 감지는 조금 더 대략적으로 처리된다.")

            total_episodes_done = train_episodes_fast(chaser_ai, runner_ai, turbo_episodes, total_episodes_done)
            print(f"\n누적 학습 게임 수: {total_episodes_done}")
            print(f"누적 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        fast_episodes = command_to_fast_episodes(command)
        if fast_episodes is not None:
            total_episodes_done = train_episodes_fast(chaser_ai, runner_ai, fast_episodes, total_episodes_done)
            print(f"\n누적 학습 게임 수: {total_episodes_done}")
            print(f"누적 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            continue

        episodes = command_to_episodes(command)
        if episodes is not None:
            print(f"\n{episodes}판을 시각화하면서 학습한다.")
            print("GUI 창이 열리면 술래와 도망자의 실제 학습 움직임이 보인다.")
            print("창을 닫으면 현재까지 진행된 학습 데이터가 저장된다.")
            total_episodes_done = train_episodes_visual(chaser_ai, runner_ai, episodes, total_episodes_done)
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
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
        elif command == "quit":
            save_learning_data(chaser_ai, runner_ai, total_episodes_done)
            print("프로그램을 종료한다.")
            break
        else:
            print("알 수 없는 명령어다.")


if __name__ == "__main__":
    main()
