import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random

class Matrixgame:
    def __init__(self):
        self.n_agents = 2
        self.n_actions = 3
        self.episode_limit = 1
        self.payoff_matrix = np.array([
            [8, -12, -12],
            [-12, 0, 0],
            [-12, 0, 0]
        ])
        self.state = np.ones(5, dtype=np.float32)

    def reset(self):
        self.state = np.ones(5, dtype=np.float32)
        return self.state, self.state

    def step(self, actions):
        reward = self.payoff_matrix[actions[0], actions[1]]
        terminated = True
        info = {"episode_limit": False}
        return reward, terminated, info

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Memory:
    def __init__(self, agent_num, action_dim):
        self.agent_num = agent_num
        self.action_dim = action_dim
        self.clear()

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = []

    def get(self):
        actions = torch.tensor(self.actions, dtype=torch.long)
        observations = torch.stack(self.observations)
        pi = [torch.stack(self.pi[i]) for i in range(self.agent_num)]
        reward = torch.tensor(self.reward, dtype=torch.float32).unsqueeze(-1)
        done = torch.tensor(self.done, dtype=torch.float32).unsqueeze(-1)
        return actions, observations, pi, reward, done


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class CriticWithVDN(nn.Module):
    def __init__(self, state_dim, action_dim, agent_num):
        super(CriticWithVDN, self).__init__()
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Update to include action_dim in the input size
        self.fc1 = nn.Linear(state_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 1)  # Output a single Q-value

    def forward(self, states, actions):
        batch_size = states.size(0)
        agent_qs = []
        
        # For each agent, concatenate its state and action
        for i in range(self.agent_num):
            agent_state = states[:, i, :]
            agent_action = actions[:, i]
            
            # One-hot encode the actions
            agent_action_one_hot = F.one_hot(agent_action, num_classes=self.action_dim).float()
            
            # Concatenate state and one-hot encoded action
            agent_input = torch.cat([agent_state, agent_action_one_hot], dim=-1)
            
            hidden = F.relu(self.fc1(agent_input))
            q_value = self.fc2(hidden)
            
            agent_qs.append(q_value)
        
        agent_qs = torch.stack(agent_qs, dim=1)
        mixed_qvals = torch.sum(agent_qs, dim=1, keepdim=True)
        
        return agent_qs, mixed_qvals


class COMA:
    def __init__(self, agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size, epsilon_start, epsilon_end, epsilon_decay):
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_steps = target_update_steps
        self.memory = Memory(agent_num, action_dim)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        
        self.actors = [Actor(state_dim, action_dim) for _ in range(agent_num)]
        self.critic = CriticWithVDN(state_dim, action_dim, agent_num)
        self.critic_target = CriticWithVDN(state_dim, action_dim, agent_num)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actors_optimizer = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        
        self.count = 0
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

    def get_actions(self, observations):
        observations = torch.tensor(np.array(observations), dtype=torch.float32)
        actions = []
        for i in range(self.agent_num):
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
                dist = self.actors[i](observations[i]).detach()
            else:
                dist = self.actors[i](observations[i]).detach()
                action = Categorical(dist).sample().item()
            self.memory.pi[i].append(dist)
            actions.append(action)
        self.memory.observations.append(observations)
        self.memory.actions.append(actions)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.total_steps * self.epsilon_decay)
        self.total_steps += 1
        
        return actions


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).view(self.batch_size, self.agent_num, -1)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).view(self.batch_size, self.agent_num, -1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for _ in range(10):  # Increase training iterations
            # Critic update
            agent_qs, mixed_qvals = self.critic(states, actions)
            critic_loss = F.mse_loss(mixed_qvals, rewards.sum(dim=1, keepdim=True))
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()
            
            # Actor update
            agent_qs, mixed_qvals = self.critic(states, actions)
            mixed_qvals = mixed_qvals.detach()
            
            for i in range(self.agent_num):
                dist = self.actors[i](states[:, i])
                log_pi = torch.log(dist + 1e-8)
                
                baseline = torch.sum(dist.detach() * mixed_qvals, dim=1, keepdim=True)
                advantage = mixed_qvals - baseline
                actor_loss = -(log_pi * advantage.detach()).sum(dim=1).mean()
                
                self.actors_optimizer[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1)
                self.actors_optimizer[i].step()
        
        if self.count % self.target_update_steps == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
        self.count += 1



def run_experiment(lr_c, lr_a, gamma, target_update_steps, total_timesteps, buffer_capacity, batch_size):
    env = Matrixgame()
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = (epsilon_start - epsilon_end) / total_timesteps
    agent = COMA(env.n_agents, env.state.shape[0], env.n_actions, lr_c, lr_a, gamma, target_update_steps, buffer_capacity, batch_size, epsilon_start, epsilon_end, epsilon_decay)
    
    joint_action_count = np.zeros((3, 3), dtype=int)
    
    for step in range(total_timesteps):
        obs, _ = env.reset()
        actions = agent.get_actions([obs, obs])
        reward, _, _ = env.step(actions)
        joint_action_count[actions[0], actions[1]] += 1
        
        # Store experience in replay buffer
        agent.replay_buffer.push([obs, obs], actions, reward, [obs, obs], True)  # Using obs as next_state since it's a one-step game
        
        if len(agent.replay_buffer) >= batch_size:
            agent.train()
        
        if (step + 1) % 1000 == 0:
            print(f"Timestep {step + 1}/{total_timesteps}")
    
    return joint_action_count

# Hyperparameters
lr_c = 0.001
lr_a = 0.0001
gamma = 0.99
target_update_steps = 500
total_timesteps = 500000  
buffer_capacity = 10000
batch_size = 64

joint_action_count = run_experiment(lr_c, lr_a, gamma, target_update_steps, total_timesteps, buffer_capacity, batch_size)

# Create and print the result matrix
env = Matrixgame()
result_matrix = np.zeros((3, 3), dtype=object)
for i in range(3):
    for j in range(3):
        reward = env.payoff_matrix[i, j]
        count = joint_action_count[i, j]
        result_matrix[i, j] = f"{reward}*{count}"

print("Joint Action Selection Results:")
print(result_matrix)