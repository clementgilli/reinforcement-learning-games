from a3c import *

NB_EPISODES = 10

file = 0
if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No file given")
assert isinstance(file,str)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

env = gym.make('ALE/Breakout-v5', render_mode='human', frameskip=1)
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
env = FrameStack(env, 4)

input_shape = env.observation_space.shape
num_actions = env.action_space.n

network = A3CNet(input_shape, num_actions).to(device)
network.load_state_dict(torch.load(file, map_location=device))
network.eval()

def select_action(net, state, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    probs, _ = net(state)
    action = torch.multinomial(probs, 1).item()
    return action

def play():
    for episode in range(NB_EPISODES):
        state = env.reset()[0]
        done = False
        local_rew = []
        while not done:
            env.render()  # Render the environment
            action = select_action(network, state, device)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            local_rew.append(reward)
        
        print(f'Episode {episode}, Rewards: {sum(local_rew)}')

play()
env.close()
