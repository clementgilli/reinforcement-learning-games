from a2c import *

NB_EPISODES = 10001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = gym.make('ALE/Breakout-v5', render_mode=None)
env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True)
env = FrameStack(env, 4)

network = A2C(env.observation_space.shape[0], env.action_space.n, device)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE,eps=1e-3)

memory = Memory()
rewards_per_episode = []
for episode in range(NB_EPISODES):
    state = env.reset()[0]
    done = False
    local_rew = []
    while not done:
        action = select_action(network, state)
        next_state, reward, done, terminated, _ = env.step(action)
        memory.add(state, action, reward, next_state, terminated)
        local_rew.append(reward)
        state = next_state

        if memory.size >= BATCH_SIZE:
            learn(network,memory,optimizer)
            memory.clear()

    rewards_per_episode.append(sum(local_rew))
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

    if mean_rewards > 100:
        print(f'Solved in {episode} episodes')
        break

    if episode % 100 == 0:
        print(f'Episode {episode}, Mean Rewards: {mean_rewards}')
        torch.save(network.state_dict(), 'net.pth')

torch.save(network.state_dict(), 'net.pth')
env.close()