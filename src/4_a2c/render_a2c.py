from a2c import *

NB_EPISODES = 10

file = 0
if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No file given")
assert isinstance(file,str)

device = torch.device('cpu')

env = gym.make('ALE/Breakout-v5', render_mode="human")
env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True)
env = FrameStack(env, 4)

network = A2C(env.observation_space.shape[0], env.action_space.n, device)
network.load_state_dict(torch.load(file,map_location=torch.device('cpu')))


for episode in range(NB_EPISODES):
    state = env.reset()[0]
    done = False
    local_rew = []
    while not done:
        action = select_action(network, state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state

    print(f"Episode : {episode}, Rewards : {sum(local_rew)}")
