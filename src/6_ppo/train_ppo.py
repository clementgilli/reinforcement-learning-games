from ppo import *


env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
n_episodes = 10000

SKIP_FRAMES = 0
refresh = 32
batch_size = 32
n_epochs = 4
alpha = 0.000025

agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    
n_games = 10000

best_score = env.reward_range[0]
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0

try:

    for i in range(n_games):
        observation,info = env.reset()
        terminated = False
        score = 0
        nbr_lives = 5
        n_steps = 0
        dead = True  
        while not terminated:
            if dead:
                observation_ = env.step(1)[0]
                dead = False
            else:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, terminated, truncated ,info = env.step(action)
                if nbr_lives > int(info["lives"]):
                    nbr_lives = int(info["lives"])
                    dead = True
                #    reward -= 1
                #reward = np.clip( reward, a_min=-1, a_max=1 )
                n_steps += 1
                score += reward
                agent.remember(observation, action, prob, val, reward, terminated)
                if n_steps % refresh == 0:
                    agent.learn()
                    learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if i%32==0:
            text = "_checkpoint_"+str(0)+".pth"
            T.save(agent.actor.state_dict(),"./saves/save_actor"+text)
            T.save(agent.critic.state_dict(),"./saves/save_critic"+text)

        if i%10==0:
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', n_steps, 'learning_steps', learn_iters)

except KeyboardInterrupt:
    print("Training interrupted by user")
    T.save(agent.actor.state_dict(),"./save_actor_interrupt.pth")
    T.save(agent.critic.state_dict(),"./save_critic_interrupt.pth")
    print("Models saved")
    env.close()
    exit()
