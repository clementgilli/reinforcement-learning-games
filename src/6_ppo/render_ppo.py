from ppo import *
from time import sleep

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
refresh = 20
batch_size = 30
n_epochs = 4
alpha = 0.0003
load = False

agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)

agent.actor.load_state_dict(T.load("save_actor_checkpoint_1340.pth",map_location=T.device('cpu')))
agent.critic.load_state_dict(T.load("save_critic_checkpoint_1340.pth",map_location=T.device('cpu')))
    
n_games = 1000

best_score = env.reward_range[0]
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0

if load:
    agent.Q_eval.load_state_dict(T.load('./saves/save_render.pth', map_location=T.device('cuda:2')))


for i in range(n_games):
    observation = env.reset()[0]
    terminated = False
    score = 0
    n_steps = 0
    nbr_lives = 5
    observation = env.step(1)[0]
    while not terminated and n_steps<500:
        action, prob, val = agent.choose_action(observation)
        print(action)
        new_observation, reward, terminated, _, info =  env.step(action)
        if nbr_lives > int(info["lives"]):
            nbr_lives = int(info["lives"])
            observation = new_observation
        n_steps += 1
        score += reward
        sleep(1/60)
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

        #if avg_score > best_score:
        #    best_score = avg_score
        #    agent.save_models()

    if i%30==0:
        text = "_checkpoint_"+str(i)+".pth"
        T.save(agent.actor.state_dict(),"./saves/save_actor"+text)
        T.save(agent.critic.state_dict(),"./saves/save_critic"+text)

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)