import random
from collections import defaultdict

# =========================================================
# 5 x 5 술래잡기 강화학습 시뮬레이션
# - 술래 AI: 도망자를 잡으면 승리
# - 도망자 AI: MAX_TURNS 동안 잡히지 않으면 승리
# - 학습 방식: Q-learning
# =========================================================

GRID_SIZE = 5
MAX_TURNS = 20
EPISODES = 30000

# Q-learning 하이퍼파라미터
ALPHA = 0.15        # 학습률: 새로 배운 값을 얼마나 반영할지
GAMMA = 0.95        # 할인율: 미래 보상을 얼마나 중요하게 볼지
EPSILON_START = 1.0 # 처음에는 랜덤 행동을 많이 하도록 설정
EPSILON_END = 0.05  # 학습 후반에도 약간은 탐험하게 설정

# 행동: 위, 아래, 왼쪽, 오른쪽
ACTIONS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class QLearningAgent:
    def __init__(self, name):
        self.name = name
        self.q_table = defaultdict(lambda: [0.0 for _ in ACTIONS])
        self.score = 0

    def choose_action(self, state, epsilon):
        """
        epsilon-greedy 방식으로 행동 선택
        - epsilon 확률: 랜덤 행동
        - 1 - epsilon 확률: 현재 Q값이 가장 높은 행동
        """
        if random.random() < epsilon:
            return random.randrange(len(ACTIONS))

        q_values = self.q_table[state]
        max_q = max(q_values)

        # 같은 최대값이 여러 개면 랜덤으로 하나 선택
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


def clamp_position(row, col):
    """격자 밖으로 나가지 못하게 좌표를 제한한다."""
    row = max(0, min(GRID_SIZE - 1, row))
    col = max(0, min(GRID_SIZE - 1, col))
    return row, col


def move(position, action_index):
    """현재 위치에서 선택한 행동대로 1칸 이동한다."""
    row, col = position
    dr, dc = ACTIONS[action_index]
    return clamp_position(row + dr, col + dc)


def random_empty_positions():
    """술래와 도망자의 시작 위치를 서로 다르게 랜덤 설정한다."""
    chaser = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    runner = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))

    while runner == chaser:
        runner = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))

    return chaser, runner


def make_state(chaser_pos, runner_pos):
    """
    상태 표현
    상태 = (술래 행, 술래 열, 도망자 행, 도망자 열)
    """
    return chaser_pos[0], chaser_pos[1], runner_pos[0], runner_pos[1]


def distance(pos1, pos2):
    """맨해튼 거리 계산: 격자에서 두 칸 사이의 거리"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def play_one_episode(chaser_ai, runner_ai, epsilon, training=True):
    """
    게임 1판 실행
    training=True이면 Q-table을 업데이트한다.
    """
    chaser_pos, runner_pos = random_empty_positions()

    for turn in range(1, MAX_TURNS + 1):
        state = make_state(chaser_pos, runner_pos)
        old_distance = distance(chaser_pos, runner_pos)

        chaser_action = chaser_ai.choose_action(state, epsilon)
        runner_action = runner_ai.choose_action(state, epsilon)

        new_chaser_pos = move(chaser_pos, chaser_action)
        new_runner_pos = move(runner_pos, runner_action)
        next_state = make_state(new_chaser_pos, new_runner_pos)
        new_distance = distance(new_chaser_pos, new_runner_pos)

        # 기본 보상은 작게 설정한다.
        # 술래는 가까워질수록 약간 보상, 멀어질수록 약간 벌점
        # 도망자는 반대로 멀어질수록 약간 보상
        chaser_reward = -0.01
        runner_reward = -0.01

        if new_distance < old_distance:
            chaser_reward += 0.05
            runner_reward -= 0.05
        elif new_distance > old_distance:
            chaser_reward -= 0.05
            runner_reward += 0.05

        winner = None

        # 술래가 도망자를 잡은 경우
        if new_chaser_pos == new_runner_pos:
            chaser_reward += 10.0
            runner_reward -= 10.0
            winner = "chaser"

        # 도망자가 제한 턴까지 버틴 경우
        elif turn == MAX_TURNS:
            chaser_reward -= 10.0
            runner_reward += 10.0
            winner = "runner"

        if training:
            chaser_ai.update_q(state, chaser_action, chaser_reward, next_state)
            runner_ai.update_q(state, runner_action, runner_reward, next_state)

        chaser_pos = new_chaser_pos
        runner_pos = new_runner_pos

        if winner is not None:
            if winner == "chaser":
                chaser_ai.score += 1
                runner_ai.score -= 1
            else:
                runner_ai.score += 1
                chaser_ai.score -= 1

            return winner, turn

    return "runner", MAX_TURNS


def train(chaser_ai, runner_ai):
    """전체 학습 실행"""
    chaser_wins = 0
    runner_wins = 0

    for episode in range(1, EPISODES + 1):
        # episode가 진행될수록 epsilon 감소
        progress = episode / EPISODES
        epsilon = EPSILON_START * (1 - progress) + EPSILON_END * progress

        winner, _ = play_one_episode(chaser_ai, runner_ai, epsilon, training=True)

        if winner == "chaser":
            chaser_wins += 1
        else:
            runner_wins += 1

        if episode % 3000 == 0:
            print(f"Episode {episode:5d} | 술래 승: {chaser_wins:5d} | 도망자 승: {runner_wins:5d} | epsilon: {epsilon:.3f}")

    print("\n학습 완료!")
    print(f"최종 점수 - 술래 AI: {chaser_ai.score}, 도망자 AI: {runner_ai.score}")


def print_grid(chaser_pos, runner_pos):
    """현재 격자 상태 출력"""
    for r in range(GRID_SIZE):
        row_text = []
        for c in range(GRID_SIZE):
            if (r, c) == chaser_pos and (r, c) == runner_pos:
                row_text.append("X")
            elif (r, c) == chaser_pos:
                row_text.append("C")  # Chaser
            elif (r, c) == runner_pos:
                row_text.append("R")  # Runner
            else:
                row_text.append(".")
        print(" ".join(row_text))
    print()


def demo_game(chaser_ai, runner_ai):
    """학습된 AI끼리 실제 게임 1판을 매턴 자세히 보여주기"""
    chaser_pos, runner_pos = random_empty_positions()

    print("\n===== 학습된 AI 시연 =====")
    print("C = 술래, R = 도망자")
    print("각 턴마다 술래 위치, 도망자 위치, 행동, 거리 변화를 출력한다.\n")

    print("[시작 상태]")
    print(f"술래 위치: {chaser_pos}, 도망자 위치: {runner_pos}")
    print(f"현재 거리: {distance(chaser_pos, runner_pos)}")
    print_grid(chaser_pos, runner_pos)

    for turn in range(1, MAX_TURNS + 1):
        state = make_state(chaser_pos, runner_pos)
        old_chaser_pos = chaser_pos
        old_runner_pos = runner_pos
        old_distance = distance(chaser_pos, runner_pos)

        # 시연에서는 epsilon=0으로 두어 학습된 최선의 행동만 선택
        chaser_action = chaser_ai.choose_action(state, epsilon=0.0)
        runner_action = runner_ai.choose_action(state, epsilon=0.0)

        chaser_pos = move(chaser_pos, chaser_action)
        runner_pos = move(runner_pos, runner_action)
        new_distance = distance(chaser_pos, runner_pos)

        print(f"===== Turn {turn} =====")
        print(f"술래   : {old_chaser_pos} -> {chaser_pos} / 행동: {ACTION_NAMES[chaser_action]}")
        print(f"도망자 : {old_runner_pos} -> {runner_pos} / 행동: {ACTION_NAMES[runner_action]}")
        print(f"거리 변화: {old_distance} -> {new_distance}")

        if new_distance < old_distance:
            print("상황 해석: 술래가 도망자에게 가까워졌다.")
        elif new_distance > old_distance:
            print("상황 해석: 도망자가 술래와의 거리를 벌렸다.")
        else:
            print("상황 해석: 두 AI 사이의 거리는 그대로다.")

        print_grid(chaser_pos, runner_pos)

        if chaser_pos == runner_pos:
            print(f"결과: 술래 승리! {turn}턴 만에 도망자를 잡았다.")
            return

    print(f"결과: 도망자 승리! {MAX_TURNS}턴 동안 잡히지 않고 버텼다.")


def evaluate(chaser_ai, runner_ai, games=1000):
    """학습된 AI의 승률 평가"""
    chaser_wins = 0
    runner_wins = 0
    total_turns = 0

    for _ in range(games):
        winner, turns = play_one_episode(chaser_ai, runner_ai, epsilon=0.0, training=False)
        total_turns += turns

        if winner == "chaser":
            chaser_wins += 1
        else:
            runner_wins += 1

    print("\n===== 평가 결과 =====")
    print(f"평가 게임 수: {games}")
    print(f"술래 승률: {chaser_wins / games * 100:.1f}%")
    print(f"도망자 승률: {runner_wins / games * 100:.1f}%")
    print(f"평균 게임 길이: {total_turns / games:.2f}턴")


if __name__ == "__main__":
    random.seed(42)

    chaser_ai = QLearningAgent("술래 AI")
    runner_ai = QLearningAgent("도망자 AI")

    train(chaser_ai, runner_ai)
    evaluate(chaser_ai, runner_ai, games=1000)
    demo_game(chaser_ai, runner_ai)
