from a3c import *

device = torch.device('cuda:1')
# WORKERS = mp.cpu_count()
WORKERS = 20

if __name__ == "__main__":
    mp.set_start_method('spawn')
    env_name = "ALE/Breakout-v5"
    env = gym.make(env_name, frameskip=1)
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, num_stack=4)

    global_net = A3CNet(env.observation_space.shape, env.action_space.n).to(device)
    global_net.share_memory()
    optimizer = optim.Adam(global_net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    global_ep, global_ep_r = mp.Value('i', 0), mp.Value('d', 0.)
    res_queue = mp.Queue()
    lock = mp.Lock()

    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, res_queue, env_name, input_shape, num_actions, DEVICE, lock) for _ in range(WORKERS)]

    [w.start() for w in workers]
    [w.join() for w in workers]

    res = []
    while not res_queue.empty():
        res.append(res_queue.get())
    print(res)
