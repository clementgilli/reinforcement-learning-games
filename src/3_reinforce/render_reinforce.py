from reinforce import *

NB_EPISODES = 2001

file = 0
if len(sys.argv) > 1:
    file = sys.argv[1]
else:
    print("No file given")
assert isinstance(file,str)

env = gym.make("CartPole-v1",render_mode="human")

net = NN(env.observation_space.shape[0], env.action_space.n)
net.load_state_dict(torch.load(file,map_location=torch.device('cpu')))

if torch.cuda.is_available():
    print("Running on GPU :", torch.cuda.get_device_name(2))
else:
    print("Running on CPU")


for i in range(NB_EPISODES):
    rewards = []
    done = False
    s = env.reset()[0]
    while not done:
        action = net(torch.tensor(np.array([s])).float()).sample().item()
        ns, r, done, terminated, _ = env.step(action)
        rewards.append(r)
        if done or sum(rewards) > 1200:
            break
        s = ns
    
    print("Episode: ", i, "Reward: ", sum(rewards))