import gymnasium as gym
import numpy as np
import pickle

# Q 테이블 로드
with open('/root/week1/YBIGTA_DS_Summer2024/Week_1/Black_Jack/Agent/Robocar_Poli.pkl', 'rb') as f:
    q_table_agent1 = pickle.load(f)

with open('/root/week1/YBIGTA_DS_Summer2024/Week_1/Black_Jack/Agent/Weak_James_Bond.pkl', 'rb') as f:
    q_table_agent2 = pickle.load(f)

# 환경 설정
env = gym.make("Blackjack-v1", natural=False)

def choose_action(q_table, state):
    return np.argmax(q_table[state])

def play_game(env, q_table):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = choose_action(q_table, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = next_state

    return total_reward

# 경쟁
num_games = 1000
agent1_wins = 0
agent2_wins = 0
draws = 0

for _ in range(num_games):
    reward_agent1 = play_game(env, q_table_agent1)
    reward_agent2 = play_game(env, q_table_agent2)

    if reward_agent1 > reward_agent2:
        agent1_wins += 1
    elif reward_agent1 < reward_agent2:
        agent2_wins += 1
    else:
        draws += 1

print(f"Agent 1 wins: {agent1_wins}")
print(f"Agent 2 wins: {agent2_wins}")
print(f"Draws: {draws}")

env.close()
