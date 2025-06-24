import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Constants
ACTIONS = (0, 1, 2, 3)
ACTION_TO_PAIR = ((-1, 0), (1, 0), (0, -1), (0, 1))

def move(start: Tuple[int, int], action: int, size: Tuple[int, int]) -> Tuple[int, int]:
    dir = ACTION_TO_PAIR[action]
    result = start[0] + dir[0], start[1] + dir[1]
    if not (0 <= result[0] < size[0] and 0 <= result[1] < size[1]):
        return start
    return result

def is_adjacent(p: Tuple[int, int], q: Tuple[int, int]) -> bool:
    return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1  # Manhattan distance of 1

@dataclass
class Env:
    pred_locs: List[Tuple[int, int]]
    prey_locs: List[Tuple[int, int]]
    size: Tuple[int, int]
    active_predators: List[bool]
    active_prey: List[bool]

    def transition(self, pred_actions: List[int], prey_actions: List[int]) -> None:
        for i, (action, active) in enumerate(zip(pred_actions, self.active_predators)):
            if active:
                self.pred_locs[i] = move(self.pred_locs[i], action, self.size)
        for i, (action, active) in enumerate(zip(prey_actions, self.active_prey)):
            if active:
                self.prey_locs[i] = move(self.prey_locs[i], action, self.size)

    def reward_and_done(self) -> Tuple[List[float], bool]:
        pred_rewards = [0.0] * len(self.pred_locs)
        
        for i, (prey_loc, prey_active) in enumerate(zip(self.prey_locs, self.active_prey)):
            if not prey_active:
                continue
            adjacent_preds = [j for j, (pred_loc, pred_active) in enumerate(zip(self.pred_locs, self.active_predators)) 
                              if pred_active and is_adjacent(prey_loc, pred_loc)]
            if len(adjacent_preds) >= 2:
                for j in adjacent_preds[:2]:  # Only the first two predators get the reward
                    pred_rewards[j] += 10.0
                    self.active_predators[j] = False
                self.active_prey[i] = False
            else:
                for j in adjacent_preds:
                    pred_rewards[j] += -0.1
        
        return pred_rewards, False  

    def get_state(self) -> List[float]:
        state = []
        for pred, active in zip(self.pred_locs, self.active_predators):
            if active:
                state.extend(pred)
            else:
                state.extend((-1, -1))  # Use (-1, -1) to represent eliminated predators
        for prey, active in zip(self.prey_locs, self.active_prey):
            if active:
                state.extend(prey)
            else:
                state.extend((-1, -1))  # Use (-1, -1) to represent eliminated prey
        return state

    def reset(self):
        rows, cols = self.size
        num_predators = len(self.pred_locs)
        num_preys = len(self.prey_locs)
        self.pred_locs = [(random.randrange(rows), random.randrange(cols)) for _ in range(num_predators)]
        self.prey_locs = [(random.randrange(rows), random.randrange(cols)) for _ in range(num_preys)]
        self.active_predators = [True] * num_predators
        self.active_prey = [True] * num_preys
        return self.get_state()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class CriticWithVDN(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(CriticWithVDN, self).__init__()
        self.num_agents = num_agents
        
        # Network to process state and action together
        self.fc1 = nn.Linear(state_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_agents)  # Output Q-values for each agent

    def forward(self, state, action):
        batch_size = state.size(0)
        
        # Concatenate state and action
        x = torch.cat((state, action), dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        individual_q_values = self.fc3(x)
        total_q_value = individual_q_values.sum(dim=1, keepdim=True)
        
        return individual_q_values, total_q_value


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class COMAAgent:
    def __init__(self, state_dim, action_dim, num_agents, lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size, epsilon_start, epsilon_end, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.batch_size = batch_size

        # Epsilon decay parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actor = Actor(state_dim, action_dim)
        self.critic = CriticWithVDN(state_dim, action_dim, num_agents)
        self.critic_target = CriticWithVDN(state_dim, action_dim, num_agents)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.count = 0
        self.total_steps = 0  # To keep track of total steps for epsilon decay

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
        
        # Implement epsilon-greedy exploration with decaying epsilon
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            return action, dist.squeeze()
        else:
            action = Categorical(dist).sample()
            return action.item(), dist.squeeze()
        
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # One-hot encode actions for the Critic
        action_one_hot = torch.zeros(states.size(0), self.action_dim)
        action_one_hot.scatter_(1, actions, 1)

        # Get individual Q-values and total Q-value from Critic
        individual_Q_values, Q_total = self.critic(states, action_one_hot)

        _, next_Q_total = self.critic_target(next_states, action_one_hot)
        next_Q_total = next_Q_total.detach()

        Q_targets = rewards + self.gamma * next_Q_total * (1 - dones)
        
        critic_loss = F.mse_loss(Q_total, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        dists = self.actor(states)
        log_probs = torch.log(dists.gather(1, actions) + 1e-10)  # Add small constant to prevent log(0)
        advantages = Q_targets - Q_total.detach()
        actor_loss = -(log_probs * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.total_steps * self.epsilon_decay)
        self.total_steps += 1

        return Q_total.mean().item()


def train(env, pred_agents, prey_agents, total_timesteps, seed, max_episode_length=200):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_rewards = []
    episode_lengths = []

    for t in range(1, total_timesteps + 1):
        pred_actions = [agent.get_action(state)[0] if env.active_predators[i] else 0 for i, agent in enumerate(pred_agents)]
        prey_actions = [random.randint(0, 3) if env.active_prey[i] else 0 for i in range(len(prey_agents))]

        env.transition(pred_actions, prey_actions)
        next_state = env.get_state()
        pred_rewards, _ = env.reward_and_done()

        episode_reward += sum(pred_rewards)
        episode_length += 1

        # Explicit episode termination conditions
        done = False
        if not any(env.active_predators):  # All predators are eliminated
            done = True
        elif not any(env.active_prey):  # All prey are caught
            done = True
        elif episode_length >= max_episode_length:  # Maximum episode length reached
            done = True

        # Update each predator agent
        for i, agent in enumerate(pred_agents):
            if env.active_predators[i]:
                action_one_hot = torch.zeros(len(ACTIONS))
                action_one_hot[pred_actions[i]] = 1.0
                agent.update(state, pred_actions[i], pred_rewards[i], next_state, done)

        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            state = env.reset()
            episode_reward = 0
            episode_length = 0

        if t % 100 == 0 or t == total_timesteps:
            avg_reward = np.mean(episode_rewards) if episode_rewards else None
            avg_length = np.mean(episode_lengths) if episode_lengths else None
            yield t, avg_reward, avg_length
            episode_rewards = []
            episode_lengths = []

        if t % 1000 == 0:
            print('time step: {}/{}'.format(t+total_timesteps*seed, total_timesteps*num_seeds))

def run_experiment(hyperparams, total_timesteps, num_seeds, epsilon_start, epsilon_end, epsilon_decay):
    rows, cols = 10, 10
    num_predators, num_preys = 6, 3
    env = Env([(random.randrange(rows), random.randrange(cols)) for _ in range(num_predators)],
              [(random.randrange(rows), random.randrange(cols)) for _ in range(num_preys)],
              (rows, cols),
              [True] * num_predators,
              [True] * num_preys)

    state_dim = (num_predators + num_preys) * 2
    action_dim = len(ACTIONS)

    lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size = hyperparams

    results = []

    for seed in range(num_seeds):
        pred_agents = [COMAAgent(state_dim, action_dim, num_predators, lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size, epsilon_start, epsilon_end, epsilon_decay) for _ in range(num_predators)]
        prey_agents = [None for _ in range(num_preys)]  # Placeholder for prey agents

        seed_results = list(train(env, pred_agents, prey_agents, total_timesteps, seed))
        results.append(seed_results)

    return results

def process_results(results):
    timesteps = [r[0] for r in results[0]]
    pred_rewards = []
    episode_lengths = []
    cumulative_pred_rewards = []  
    cumulative_episode_lengths = []  

    for t in range(len(timesteps)):
        rewards_at_t = [r[t][1] for r in results if t < len(r) and r[t][1] is not None]
        lengths_at_t = [r[t][2] for r in results if t < len(r) and r[t][2] is not None]
        
        if rewards_at_t:
            pred_rewards.append(np.median(rewards_at_t))
        else:
            pred_rewards.append(None)
        
        if lengths_at_t:
            episode_lengths.append(np.mean(lengths_at_t))
        else:
            episode_lengths.append(None)

        all_rewards_up_to_t = [r[i][1] for r in results for i in range(min(t+1, len(r))) if r[i][1] is not None]
        all_lengths_up_to_t = [r[i][2] for r in results for i in range(min(t+1, len(r))) if r[i][2] is not None]
        
        cumulative_pred_rewards.append(np.median(all_rewards_up_to_t) if all_rewards_up_to_t else None)
        cumulative_episode_lengths.append(np.mean(all_lengths_up_to_t) if all_lengths_up_to_t else None)

    return timesteps, pred_rewards, episode_lengths, cumulative_pred_rewards, cumulative_episode_lengths 

if __name__ == "__main__":
    hyperparams = (0.001, 0.001, 0.99, 1000, 10000, 32)  # lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size
    total_timesteps = 500000
    num_seeds = 3
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = (epsilon_start - epsilon_end) / total_timesteps

    results = run_experiment(hyperparams, total_timesteps, num_seeds, epsilon_start, epsilon_end, epsilon_decay)
    timesteps, pred_rewards, episode_lengths, cumulative_pred_rewards, cumulative_episode_lengths = process_results(results)  

    # Remove None values for plotting
    valid_data = [(t, r, l, cr, cl) for t, r, l, cr, cl in zip(timesteps, pred_rewards, episode_lengths, cumulative_pred_rewards, cumulative_episode_lengths) if r is not None and l is not None and cr is not None and cl is not None]
    timesteps, pred_rewards, episode_lengths, cumulative_pred_rewards, cumulative_episode_lengths = zip(*valid_data)  

    # Convert lists to NumPy arrays
    pred_rewards_array = np.array(pred_rewards)
    episode_lengths_array = np.array(episode_lengths)
    cumulative_pred_rewards_array = np.array(cumulative_pred_rewards)  
    cumulative_episode_lengths_array = np.array(cumulative_episode_lengths)  

    # Save as .npy files
    np.save('pred_rewards_coma_vdn.npy', pred_rewards_array)
    np.save('episode_lengths_coma_vdn.npy', episode_lengths_array)
    np.save('cumulative_pred_rewards_coma_vdn.npy', cumulative_pred_rewards_array)  
    np.save('cumulative_episode_lengths_coma_vdn.npy', cumulative_episode_lengths_array)  

    plt.figure(figsize=(12, 10))  
    
    plt.subplot(2, 2, 1)  
    plt.plot(timesteps, pred_rewards)
    plt.title("Median Predator Reward_coma_vdn")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)  
    plt.plot(timesteps, episode_lengths)
    plt.title("Average Episode Length_coma_vdn")
    plt.xlabel("Timesteps")
    plt.ylabel("Length")

    plt.subplot(2, 2, 3)
    plt.plot(timesteps, cumulative_pred_rewards)
    plt.title("Cumulative Median Predator Reward_coma_vdn")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Reward")

    plt.subplot(2, 2, 4)
    plt.plot(timesteps, cumulative_episode_lengths)
    plt.title("Cumulative Average Episode Length_coma_vdn")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Length")

    plt.tight_layout()
    plt.show()
                                          