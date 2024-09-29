import FirstVisitMonteCarlos as fvmc
import numpy as np
import gym

env = gym.make('Blackjack-v1', render_mode=None)
agent = fvmc.FirstVisitMonteCarlo(env, gamma=0.9, epsilon=0.1)
agent.train(2000000)
agent.save_Q("./")
# agent.load_Q("./")
agent.play(1000)
agent.plot_value_function_BJ()