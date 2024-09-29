import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import sys

GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
CLIP_GRAD = 0.1

class A2C(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(A2C, self).__init__()

        self.common = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        self.actor = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.common.to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.device = device

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)

class Memory:
    def __init__(self):
        self.states = []
        self.snd_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.snd_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.size += 1

    def load(self):
        return self.states, self.snd_states, self.actions, self.rewards, self.dones

    def clear(self):
        self.states = []
        self.snd_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.size = 0

def select_action(net, state):
    device = net.device
    state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
    probs, _ = net(state)
    action = torch.multinomial(probs, 1).item()
    return action

def learn(network,memory,optimizer):
    device = network.device
    states,snd_states,actions,rewards,dones = memory.load()
    states_v = torch.FloatTensor(np.array(states)).to(device)
    next_states_v = torch.FloatTensor(np.array(snd_states)).to(device)
    actions_t = torch.LongTensor(np.array(actions)).to(device)
    rewards_v = torch.FloatTensor(np.array(rewards)).view(-1).to(device)
    dones_v = torch.FloatTensor(np.array(dones)).to(device)

    policy,values = network(states_v)
    values = values.view(-1)
    _, next_values = network(next_states_v)
    next_values = next_values.view(-1)

    log_probs = torch.log(policy[range(len(actions)), actions_t])

    entropy = -torch.sum(policy * torch.log(policy), dim=-1)

    advantage = rewards_v + (1-dones_v)*GAMMA*next_values - values

    actor_loss = -torch.sum(log_probs * advantage.detach()) - ENTROPY_BETA * torch.sum(entropy)

    critic_loss = torch.sum(advantage.pow(2))

    optimizer.zero_grad()
    total_loss = actor_loss + critic_loss
    #actor_loss.backward(retain_graph=True)
    #critic_loss.backward()
    total_loss.backward()
    #nn.utils.clip_grad_norm_(network.parameters(), CLIP_GRAD)
    optimizer.step()