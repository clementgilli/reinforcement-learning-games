import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch.multiprocessing as mp
import os
import sys

GAMMA = 0.99
LEARNING_RATE = 0.003
ENTROPY_BETA = 0.02
BATCH_SIZE = 32
CLIP_GRAD = 0.1
T_MAX = 5  # Steps before sync
SAVE_INTERVAL = 100  # Save every 100 epochs

class A3CNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        return self.policy(conv_out), self.value(conv_out)

class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, global_ep_r, res_queue, env_name, input_shape, num_actions, device, lock):
        super(Worker, self).__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.local_net = A3CNet(input_shape, num_actions).to(device)
        self.global_ep = global_ep
        self.global_ep_r = global_ep_r
        self.res_queue = res_queue
        self.env_name = env_name
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.env = self.create_env()
        self.lock = lock

    def create_env(self):
        env = gym.make(self.env_name, frameskip=1)
        env = AtariPreprocessing(env, scale_obs=True)
        env = FrameStack(env, num_stack=4)
        return env

    def run(self):
        total_step = 1
        while self.global_ep.value < 5000:  # Training until episode 5000
            state = self.env.reset()[0]  # Ensure state is reset properly
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                state = np.array(state)  # Ensure state is in proper format
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.array(next_state)  # Ensure next_state is in proper format
                ep_r += reward

                buffer_a.append(action)
                buffer_r.append(reward)
                buffer_s.append(state)

                if total_step % T_MAX == 0 or done:
                    self.sync()
                    self.learn(buffer_s, buffer_a, buffer_r, done)

                    if done:
                        with self.global_ep.get_lock():
                            self.global_ep.value += 1
                            if self.global_ep.value % SAVE_INTERVAL == 0:
                                torch.save(self.global_net.state_dict(), f'global_net_{self.global_ep.value}.pth')

                        if self.global_ep.value % 20 == 0:
                            print(ep_r)

                        self.global_ep_r.value = ep_r
                        self.res_queue.put(self.global_ep_r.value)
                        break
                    buffer_s, buffer_a, buffer_r = [], [], []

                state = next_state
                total_step += 1

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, _ = self.local_net(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def sync(self):
        self.local_net.load_state_dict(self.global_net.state_dict())

    def learn(self, states, actions, rewards, done):
        states_v = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_v = torch.FloatTensor(np.array(rewards)).to(self.device)

        policy, values = self.local_net(states_v)
        values = values.view(-1)
        returns = self.compute_returns(rewards_v, values, done)

        log_probs = torch.log(policy[range(len(actions)), actions_t])
        entropy = -torch.sum(policy * torch.log(policy), dim=-1)
        advantage = returns - values

        actor_loss = -torch.sum(log_probs * advantage.detach()) - ENTROPY_BETA * torch.sum(entropy)
        critic_loss = torch.sum(advantage.pow(2))

        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        nn.utils.clip_grad_norm_(self.local_net.parameters(), CLIP_GRAD)

        with self.lock:
            for global_param, local_param in zip(self.global_net.parameters(), self.local_net.parameters()):
                if local_param.grad is not None:
                    global_param._grad = local_param.grad
            self.optimizer.step()

        self.local_net.load_state_dict(self.global_net.state_dict())

    def compute_returns(self, rewards, values, done):
        returns = torch.zeros_like(rewards).to(self.device)
        R = 0 if done else values[-1].detach()
        for t in reversed(range(len(rewards))):
            R = rewards[t] + GAMMA * R
            returns[t] = R
        return returns