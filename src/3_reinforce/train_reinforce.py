from reinforce import *

NB_EPISODES = 2001

env = gym.make("CartPole-v1")
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
net = NN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

if torch.cuda.is_available():
    print("Running on GPU :", torch.cuda.get_device_name(2))
else:
    print("Running on CPU")

rewards_per_episode = []
total_mean_rewards = []

batch_states, batch_actions, batch_qvals = [], [], []
for i in range(NB_EPISODES):
    #on joue plusieurs episodes pour remplir les batchs
    rewards = []
    done = False
    s = env.reset()[0]
    while not done:
        action = net(torch.tensor(np.array([s])).to(device).float()).sample().item()
        ns, r, done, terminated, _ = env.step(action)
        rewards.append(r)
        batch_states.append(s)
        batch_actions.append(action)
        if done or sum(rewards) > 1200:
            break
        s = ns
    
    #on stocke pour avoir les graphiques
    rewards_per_episode.append(sum(rewards))
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
    total_mean_rewards.append(mean_rewards)

    #on remplit le qval batch en calculant les rewards avec le discount factor
    batch_qvals.extend(calc_qvals(rewards))
    
    if i%100 == 0:
        print("Episode: ", i, "Reward: ", sum(rewards), "Mean Reward: ", mean_rewards)

    if mean_rewards > 1000:
        print("Solved in ", i, " episodes")
        break

    if len(batch_states) < EPISODE_TO_TRAIN:
        continue
    
    #une fois le batch rempli, on continue
    learn(batch_states, batch_actions, batch_qvals, net, optimizer, device)

    #on vide les batch
    batch_states.clear()
    batch_actions.clear()
    batch_qvals.clear()

torch.save(net.state_dict(), "save.pth")

fig, axs = plt.subplots(1, 1, figsize=(16, 8))

axs.plot(total_mean_rewards)
axs.set_title('Mean Rewards')
plt.show()