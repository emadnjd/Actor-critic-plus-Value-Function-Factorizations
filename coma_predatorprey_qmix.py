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

class QMixerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, mixing_embed_dim):
        super(QMixerCritic, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = mixing_embed_dim

        self.agent_qs = nn.ModuleList([nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ) for _ in range(n_agents)])

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, state, actions):
        bs = state.size(0)
        states = state.view(bs, -1, self.state_dim)
        actions = actions.view(bs, self.n_agents, -1)

        agent_qs = []
        for i in range(self.n_agents):
            agent_input = torch.cat([states[:, i], actions[:, i]], dim=-1)
            agent_q = self.agent_qs[i](agent_input)
            agent_qs.append(agent_q)

        agent_qs = torch.stack(agent_qs, dim=1)
        
        w1 = torch.abs(self.hyper_w_1(states[:, 0]))
        b1 = self.hyper_b_1(states[:, 0])
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.transpose(1, 2), w1) + b1)
        
        w_final = torch.abs(self.hyper_w_final(states[:, 0]))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states[:, 0]).view(-1, 1, 1)
        
        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1)

        return q_tot, agent_qs.squeeze(-1)

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
    def __init__(self, state_dim, action_dim, n_agents, lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size, epsilon_start, epsilon_end, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.batch_size = batch_size
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.actor = Actor(state_dim, action_dim)
        self.critic = QMixerCritic(state_dim, action_dim, n_agents, mixing_embed_dim=32)
        self.critic_target = QMixerCritic(state_dim, action_dim, n_agents, mixing_embed_dim=32)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.count = 0
        self.total_steps = 0

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(state)
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = Categorical(dist).sample().item()
        
        return action, dist.squeeze()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Reshape tensors
        states = states.view(self.batch_size, -1)
        next_states = next_states.view(self.batch_size, -1)
        actions = actions.view(self.batch_size, 1)
        rewards = rewards.view(self.batch_size, 1)
        dones = dones.view(self.batch_size, 1)

        # Convert actions to one-hot encoding
        actions_one_hot = F.one_hot(actions.squeeze(-1), num_classes=self.action_dim).float()

        Q_tot, Q_values = self.critic(states.unsqueeze(1).repeat(1, self.n_agents, 1), actions_one_hot.unsqueeze(1).repeat(1, self.n_agents, 1))
        
        with torch.no_grad():
            next_dists = self.actor(next_states)
            next_actions = Categorical(next_dists).sample().view(self.batch_size, 1)
            next_actions_one_hot = F.one_hot(next_actions.squeeze(-1), num_classes=self.action_dim).float()
            Q_next_tot, _ = self.critic_target(next_states.unsqueeze(1).repeat(1, self.n_agents, 1), next_actions_one_hot.unsqueeze(1).repeat(1, self.n_agents, 1))
        
        Q_targets = rewards + self.gamma * Q_next_tot * (1 - dones)
        
        critic_loss = F.mse_loss(Q_tot, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        dists = self.actor(states)
        log_probs = torch.log(dists.gather(1, actions) + 1e-10)
        advantages = Q_targets - Q_tot.detach()
        actor_loss = -(log_probs * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.total_steps += 1

        return Q_tot.mean().item()

def train(env, pred_agents, num_preys, total_timesteps, seed, max_episode_length=200):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_rewards = []
    episode_lengths = []

    for t in range(1, total_timesteps + 1):
        pred_actions = [agent.get_action(state)[0] if env.active_predators[i] else 0 
                        for i, agent in enumerate(pred_agents)]
        prey_actions = [random.randint(0, 3) if env.active_prey[i] else 0 
                        for i in range(num_preys)]

        env.transition(pred_actions, prey_actions)
        next_state = env.get_state()
        pred_rewards, done = env.reward_and_done()

        episode_reward += sum(pred_rewards)
        episode_length += 1

        if not any(env.active_predators) or not any(env.active_prey) or episode_length >= max_episode_length:
            done = True
        
        for i, agent in enumerate(pred_agents):
            if env.active_predators[i]:
                agent.update(state, pred_actions[i], pred_rewards[i], next_state, done)

        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            state = env.reset()
            episode_reward = 0
            episode_length = 0

        if t % 100 == 0 or t == total_timesteps:  
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else None
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else None
            yield t, avg_reward, avg_length

        if t % 1000 == 0:
            print(f'Time step: {t}/{total_timesteps}, Seed: {seed}')

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

        seed_results = list(train(env, pred_agents, num_preys, total_timesteps, seed))
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
        
        cumulative_pred_rewards.append(np.mean(all_rewards_up_to_t) if all_rewards_up_to_t else None)
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
    np.save('pred_rewards_coma_qmix.npy', pred_rewards_array)
    np.save('episode_lengths_coma_qmix.npy', episode_lengths_array)
    np.save('cumulative_pred_rewards_coma_qmix.npy', cumulative_pred_rewards_array)
    np.save('cumulative_episode_lengths_coma_qmix.npy', cumulative_episode_lengths_array)

    plt.figure(figsize=(12, 10))  
    
    plt.subplot(2, 2, 1)  
    plt.plot(timesteps, pred_rewards)
    plt.title("Median Predator Reward_coma_qmix")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)  
    plt.plot(timesteps, episode_lengths)
    plt.title("Average Episode Length_coma_qmix")
    plt.xlabel("Timesteps")
    plt.ylabel("Length")

    # subplots for cumulative metrics
    plt.subplot(2, 2, 3)
    plt.plot(timesteps, cumulative_pred_rewards)
    plt.title("Cumulative Median Predator Reward_coma_qmix")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Reward")

    plt.subplot(2, 2, 4)
    plt.plot(timesteps, cumulative_episode_lengths)
    plt.title("Cumulative Average Episode Length_coma_qmix")
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Length")

    plt.tight_layout()
    plt.show()