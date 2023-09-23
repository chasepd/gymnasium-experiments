import gymnasium as gym
import numpy as np

# Initialize environment
env = gym.make("LunarLander-v2", render_mode="human")

# Initialize Q-tables 
n_bins = 10
state_space_size = [n_bins] * 8  # 8-dimensional state space, discretized into 10 bins each
n_actions = env.action_space.n

# Check if saved Q-tables exist, and load them if they do. Otherwise, initialize with zeros.
try:
    Q1 = np.load("Q1.npy")
    Q2 = np.load("Q2.npy")
except:
    Q1 = np.zeros(state_space_size + [n_actions])
    Q2 = np.zeros(state_space_size + [n_actions])

# Hyperparameters
alpha = 0.1
gamma = 0.99
initial_epsilon = 0.6
epsilon_decay = 0.995
min_epsilon = 0.01
epsilon = initial_epsilon

# Helper function to discretize the state space
def discretize_state(observation, n_bins=10):
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    discretized_state = []
    for dim, bounds in zip(observation, state_bounds):
        bin_width = (bounds[1] - bounds[0]) / n_bins
        discrete_value = int((dim - bounds[0]) / bin_width)
        discrete_value = min(max(discrete_value, 0), n_bins - 1)  # Clip to ensure within bounds
        discretized_state.append(discrete_value)
    return tuple(discretized_state)

# Initialize state
observation, info = env.reset()
state = discretize_state(observation)

# Performance tracking
average_rewards = []

# Double Q-learning algorithm
for episode in range(1000):  # Training for 1000 episodes
    total_reward = 0
    for _ in range(10000):  # Each episode lasts maximum 10000 steps
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q1[state] + Q2[state])  # Take action based on the sum of Q1 and Q2

        # Take action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        next_state = discretize_state(observation)
                
        if terminated or truncated:
            Q1[state][action] = (1 - alpha) * Q1[state][action] + alpha 
            Q2[state][action] = (1 - alpha) * Q2[state][action] + alpha
        else:
            best_next_action = np.argmax(Q1[state] + Q2[state])
            Q1[state][action] = (1 - alpha) * Q1[state][action] + alpha * (reward + gamma * Q2[next_state][best_next_action])
            Q2[state][action] = (1 - alpha) * Q2[state][action] + alpha * (reward + gamma * Q1[next_state][best_next_action])

        # Update state
        state = next_state
        
        if terminated or truncated:
            observation, info = env.reset()
            state = discretize_state(observation)
            break
    
    # Update epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    
    # Log performance
    average_rewards.append(total_reward)
    if episode % 10 == 0:
        print(f"Episode {episode}, Average Reward: {np.mean(average_rewards[-100:])}")
    
    # Slowly decrease alpha
    alpha *= 0.999

# Save Q-tables
np.save("Q1.npy", Q1)
np.save("Q2.npy", Q2)

# Evaluation run
observation = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
state = discretize_state(observation)
total_reward = 0
for _ in range(10000):
    action = np.argmax(Q1[state] + Q2[state])
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    state = discretize_state(observation)
    env.render()
    if terminated:
        print(f"Total reward: {total_reward}")
        break

env.close()
