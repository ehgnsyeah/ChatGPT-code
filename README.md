# 5x5 Tag Reinforcement Learning Simulation

Python으로 구현한 5x5 격자 술래잡기 강화학습 시뮬레이션입니다.

## 설명

- 술래 AI와 도망자 AI가 각각 1명씩 존재합니다.
- 각 AI는 한 턴에 한 칸씩 이동합니다.
- 술래가 도망자를 잡으면 술래 승리입니다.
- 도망자가 제한 턴 동안 버티면 도망자 승리입니다.
- Q-learning 방식으로 두 AI를 학습시킵니다.
- tag_rl_simulation_persistent_v2-1.py 의 경우 이미 84000번 가량 학습시켰습니다.

## 실행 방법

```bash
python tag_rl_simulation_qlearning.py
```

## 이동가능한 경기장
```bash
(0,500) -------------------- (500,500)
   |                            |
   |                            |
   |     C              R       |   y = 250
   |   (100,250)    (400,250)   |
   |                            |
(0,0) ---------------------- (500,0)
```
