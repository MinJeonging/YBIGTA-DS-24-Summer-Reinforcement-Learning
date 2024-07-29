import gymnasium as gym
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict, deque
import random
import optuna

# 환경 설정
env = gym.make("Blackjack-v1", natural=False)

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
def train_agent(agent_id, alpha, gamma, epsilon, epsilon_min, epsilon_decay, alpha_decay, n, memory_size, batch_size, reward_clip, num_episodes):
    q_table = initialize_q_table()
    memory = deque(maxlen=memory_size)
    
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

    return q_table

# 목적 함수 정의
def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1e-1)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.999)
    epsilon = trial.suggest_uniform('epsilon', 0.5, 1.0)
    epsilon_min = trial.suggest_uniform('epsilon_min', 0.01, 0.1)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.99995, 0.999999)
    alpha_decay = trial.suggest_uniform('alpha_decay', 0.99995, 0.999999)
    n = trial.suggest_int('n', 1, 10)
    memory_size = trial.suggest_int('memory_size', 1000, 10000)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    reward_clip = trial.suggest_uniform('reward_clip', 0.5, 2.0)
    num_episodes = 5000  # 트라이얼마다 5000 에피소드로 평가

    total_rewards = []

    for agent_id in range(num_agents):
        q_table = train_agent(agent_id, alpha, gamma, epsilon, epsilon_min, epsilon_decay, alpha_decay, n, memory_size, batch_size, reward_clip, num_episodes)
        
        # 에이전트 평가
        total_reward = 0
        for _ in range(100):  # 100번의 에피소드로 평가
            state, info = env.reset()
            done = False
            while not done:
                action = np.argmax(q_table[state])
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                state = next_state
                done = terminated or truncated

        total_rewards.append(total_reward)

    return np.mean(total_rewards)

# 하이퍼파라미터 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # 100번의 트라이얼 수행

print("Best hyperparameters: ", study.best_params)

env.close()
