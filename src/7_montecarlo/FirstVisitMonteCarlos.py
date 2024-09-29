import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import pickle as pkl

class FirstVisitMonteCarlo:
    def __init__(self, env, gamma=1.0, epsilon=1):
        self.env = env              # environment de l'entrainement
        self.gamma = gamma          # facteur de remise
        self.epsilon = epsilon      # epsilon pour la politique epsilon-greedy
        # self.returns = {}           # dictionnaire des retours
        self.Q = {}                 # dictionnaire des valeurs d'état-action
        self.policy = {}            # dictionnaire de la politique
        self.returns_count = {}     # dictionnaire du nombre de retours
        self.episode = []           # liste des états du dernier épisodes
        self.episode_reward = 0     # récompense de l'épisode

    def generate_episode(self):
        """
        :return: Liste des couples (état, action, récompense) de l'épisode.
        Génération d'un épisode en fonction de la politique actuel
        """
        state = self.env.reset()[0]
        self.episode = []
        while True:
            action = self.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.episode.append((state, action, reward))
            state = next_state
            self.episode_reward += reward
            if done:
                break
        return self.episode
    
    def choose_action(self, state):
        """
        :param state: Etat courant
        :return: Action à prendre
        """
        if state not in self.policy or np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        return self.policy[state]
    
    def update_policy(self):
        """
        Mise à jour de la politique et des valeurs d'état-action
        """
        G = 0
        visited = []
        for state, action, reward in reversed(self.episode):
            
            G = self.gamma*G + reward
            if (state, action) not in visited:
                """ if (state,action) not in self.returns:
                    self.returns[(state,action)] = []
                self.returns[(state, action)].append(G)
                self.Q[state,action] = np.mean(self.returns[(state, action)]) """
                if (state, action) not in self.Q:
                    self.Q[state, action] = 0
                    self.returns_count[(state, action)] = 0
                self.returns_count[(state, action)] += 1
                self.Q[state,action] += (G - self.Q[state,action]) / self.returns_count[(state, action)]

                # Update de la politique
                Q_max = float('-inf')
                for a in range(self.env.action_space.n) :
                    if (state,a) in self.Q and self.Q[state,a] > Q_max:
                        self.policy[state] = a
                        Q_max = self.Q[state,a]
            visited.append((state, action))            

    def save_Q(self, dir_path):
        with open(dir_path+"Q.pickle", "wb") as file:
            pkl.dump(self.Q, file, protocol=pkl.HIGHEST_PROTOCOL)
        with open(dir_path+"policy.pickle", "wb") as file:
            pkl.dump(self.policy, file, protocol=pkl.HIGHEST_PROTOCOL)

    def load_Q(self, dir_path):
        with open(dir_path+"Q.pickle", 'rb') as file:
            self.Q = pkl.load(file)
        with open(dir_path+"policy.pickle", 'rb') as file:
            self.policy = pkl.load(file)


    def train(self, num_episodes, update_info = 1000):
        """
        :param num_episodes: Nombre d'épisodes à jouer
        :return: La politique optimale
        """
        print(f"Training for {num_episodes} episodes.")
        for _ in range(num_episodes):
            self.generate_episode()
            self.epsilon = 10**(-2/num_episodes)
            self.update_policy()
            if _ % update_info == 0:
                print(f"Episode {_} done.")
        return self.policy
    
    def play(self, num_episodes):
        """
        :param num_episodes: Nombre d'épisodes à jouer
        :return: La récompense moyenne des épisodes
        """
        rewards = []
        for _ in range(num_episodes):
            self.episode_reward = 0
            self.generate_episode()
            rewards.append(self.episode_reward)
        mean_reward = np.mean(rewards)

        print(f"Mean reward of {mean_reward} out of {num_episodes} episodes.")
        return mean_reward

    def plot_value_function_BJ(self):
        """
        Plot the value function for Blackjack
        """
        player_sum = np.arange(12, 22)
        dealer_sum = np.arange(1, 11)

        value_function = -np.ones((len(player_sum), len(dealer_sum)))

        for (player, dealer,ace), action in self.Q:
            if 10<=player<=21 and 1<=dealer<=11:
                if self.Q[(player,dealer,ace), action] >value_function[player-12, dealer-1]:
                    value_function[player-12, dealer-1] = self.Q[(player,dealer,ace), action]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(dealer_sum, player_sum)
        surf = ax.plot_surface(X, Y, value_function, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel('Player Sum')
        ax.set_xlabel('Dealer Sum')
        ax.set_zlabel('Value')
        ax.set_title('Value Function')
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()