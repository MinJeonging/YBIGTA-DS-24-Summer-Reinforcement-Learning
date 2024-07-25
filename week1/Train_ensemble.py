import gymnasium as gym
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict, deque
import random

# 환경 설정
env = gym.make("Blackjack-v1", natural=False)

# Q-learning 파라미터 설정 - 학습된 최적의 하이퍼파라미터로 설정
alpha = 0.014423372736795956
gamma = 0.9306100055770548
epsilon = 0.9513314880390973
epsilon_min = 0.059714013527123536
epsilon_decay = 0.9999627349059398
alpha_decay = 0.999992427649312
num_episodes = 10000000
n = 4
memory_size = 9902
batch_size = 45
reward_clip = 1.345498571265993
num_agents = 100

# Q 테이블 초기화 함수
def initialize_q_table():
    return defaultdict(lambda: np.zeros(env.action_space.n))

# 행동 선택 함수
def choose_action(state, epsilon, q_table):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# 경험을 저장하는 함수
def store_experience(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# 경험을 샘플링하는 함수
def sample_experience(memory, batch_size):
    return random.sample(memory, batch_size)

# 보상을 클리핑하는 함수
def clip_reward(reward, clip_value):
    return max(min(reward, clip_value), -clip_value)

# 에이전트 학습 함수
def train_agent(agent_id):
    q_table = initialize_q_table()
    memory = deque(maxlen=memory_size)
    alpha = 0.1
    epsilon = 1.0

    for episode in tqdm(range(num_episodes), desc=f"Agent {agent_id} Training"):
        state, info = env.reset()
        done = False
        trajectory = deque(maxlen=n)
        rewards = deque(maxlen=n)

        while not done:
            action = choose_action(state, epsilon, q_table)
            next_state, reward, terminated, truncated, info = env.step(action)

            # 보상 클리핑
            reward = clip_reward(reward, reward_clip)

            store_experience(memory, state, action, reward, next_state, terminated or truncated)

            trajectory.append((state, action))
            rewards.append(reward)

            if len(trajectory) == n:
                state_t, action_t = trajectory.popleft()
                total_return = sum([gamma**i * rewards[i] for i in range(n)])
                if not (terminated or truncated):
                    total_return += gamma**n * np.max(q_table[next_state])
                q_table[state_t][action_t] += alpha * (total_return - q_table[state_t][action_t])

            state = next_state
            done = terminated or truncated

        # 잔여 샘플들에 대해 업데이트
        while len(trajectory) > 0:
            state_t, action_t = trajectory.popleft()
            total_return = sum([gamma**i * rewards[i] for i in range(len(rewards))])
            q_table[state_t][action_t] += alpha * (total_return - q_table[state_t][action_t])
            rewards.popleft()

        # Experience Replay에서 샘플을 뽑아 학습
        if len(memory) >= batch_size:
            experiences = sample_experience(memory, batch_size)
            for state, action, reward, next_state, done in experiences:
                if done:
                    td_target = reward
                else:
                    td_target = reward + gamma * np.max(q_table[next_state])
                td_delta = td_target - q_table[state][action]
                q_table[state][action] += alpha * td_delta

        # Epsilon과 학습률 업데이트
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        alpha *= alpha_decay

    # Q 테이블 저장
    with open(f'q_table_agent_{agent_id}.pkl', 'wb') as f:
        pickle.dump(dict(q_table), f)

    return q_table

# 에이전트 학습 및 저장
for agent_id in range(num_agents):
    train_agent(agent_id)

# Q 테이블 앙상블 함수
def ensemble_q_tables(num_agents):
    q_tables = []

    # 각 에이전트의 Q 테이블 로드
    for agent_id in range(num_agents):
        with open(f'q_table_agent_{agent_id}.pkl', 'rb') as f:
            q_tables.append(pickle.load(f))

    # 모든 Q 테이블을 합산하여 평균 계산
    ensemble_q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    for q_table in q_tables:
        for state, actions in q_table.items():
            ensemble_q_table[state] += actions

    for state in ensemble_q_table:
        ensemble_q_table[state] /= num_agents

    return ensemble_q_table

# 최적의 Q 테이블 얻기
optimal_q_table = ensemble_q_tables(num_agents)

# 최적의 Q 테이블 저장
with open('Robocar_Poli.pkl', 'wb') as f:
    pickle.dump(dict(optimal_q_table), f)

env.close()
print("Training finished and optimal Q-table saved.")
