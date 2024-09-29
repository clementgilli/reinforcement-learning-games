import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys

GAMMA = 0.99
LEARNING_RATE = 0.01
ENTROPY_BETA = 0.01
EPISODE_TO_TRAIN = 8

class NN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.softmax(x, dim=1)
        return Categorical(x)
        #ou alors juste return Categorical(logits=x) sans le softmax

def calc_qvals(rewards):
    #calcule les qvals d'un episode
    qvals = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        qvals.append(sum_r)
    return list(reversed(qvals))

def learn(batch_states, batch_actions, batch_qvals, net, optimizer, device):
    states_v = torch.FloatTensor(np.array(batch_states)).to(device)
    actions_t = torch.LongTensor(np.array(batch_actions)).to(device)
    qvals_v = torch.FloatTensor(np.array(batch_qvals)).to(device)
   
    #calcul de la policy (renvoie une densite de proba)
    logits = net(states_v)
    
    #on calcule la loss de la policy (voir cours)
    log_prob_v = logits.log_prob(actions_t)
    loss_policy_v = - GAMMA * log_prob_v * (qvals_v - qvals_v.mean()) #on soustrait la moyenne pour stabiliser l'apprentissage
    loss_policy_v = loss_policy_v.mean()

    #on calcule aussi l'entropy loss pour punir l'agent d'etre trop sur de lui
    probs_v = nn.functional.softmax(logits.logits, dim=1)
    entropy_v = - (probs_v * torch.log(probs_v)).sum(dim=1).mean()
    entropy_loss_v = -ENTROPY_BETA * entropy_v

    #on somme les 2
    loss_v = (loss_policy_v + entropy_loss_v).to(device)

    optimizer.zero_grad()
    loss_v.backward()
    optimizer.step()