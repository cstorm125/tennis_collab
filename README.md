# tennis_collab
DDPG with shared rewards to solve Tennis in a Unity ML-Toolkit Environment

`tennis_collab` is a deep deterministic policy gradeint (DDPG) agent trained to play [Unity ML-Agent Toolkit](https://github.com/Unity-Technologies/ml-agents)'s [Tennis](https://www.youtube.com/watch?v=WprTJwaK510). Instead of training the agents to *compete* against each other, we trained them using **the sum of rewards** for them to *cooperate* with each other for the longest rally possible. The model solved the environment (scoring a 100-play moving average across all agents of 0.5 or above) in 403 episodes (roughly 8 minutes). The weights of trained network are saved as `actor_optimal.pth` and `critic_optimal.pth`.

## Environment

The environment consists of 24 values that describe a state (8 values stacked over 3 timesteps) and 2 values that describe an action. The state represents the position and velocity of the ball and racket. The action is  movement toward (or away from) the net, and jumping. The episode ends after when the ball goes out of bound. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. The environment is considered solved when the average score of the last 100 episodes exceed 0.5.

```
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: (2, 24)
The state looks like for the first agent: [ 0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.         -6.65278625 -1.5
 -0.          0.          6.83172083  6.         -0.          0.        ]
```

## Getting Started

0. Install dependencies.

```
pip install -r requirements.txt
```

1. Clone this repository and install `unityagents`.

```
pip -q install ./python
```

2. Import `UnityEnvironment` and load environment.

```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Reacher_Linux_OneAgent/Reacher.x86_64")
```

3. Follow `train_agent.ipynb` to train the agent.

4. Our implementation is divided as follows:
* `memory.py` - Experience Replay Memory
* `agent.py` - Agent
* `network` - Networks for local and target

## Train Agent

These are the steps you can take to train the agent with default settings.

1. Create a experience replay memory.

```
mem = VanillaMemory(int(1e5), seed = 0)
```

2. Create an agent.

```
agent = Agent(state_size=24, action_size=2, replay_memory=mem, random_seed=0, 
              nb_agent = 2, bs = 128,
              gamma=0.99, tau = 1e-1, lr_actor=5e-4, lr_critic=1e-3, wd_actor=0, wd_critic=0,
              clip_actor = None, clip_critic = 1, update_interval = 1, update_times = 1)
```

3. Train the agent.

```
scores = []
moving_scores = []
scores_avg = deque(maxlen=100)
n_episodes = 1000
nb_agent = 2

for episode in trange(n_episodes):
    #get initial states
    env_info = env.reset(train_mode=True)[brain_name]            
    states = env_info.vector_observations
    score = np.zeros(nb_agent)
    agent.reset_noise()                                             

    while True:
        #random actions
        actions = np.random.randn(nb_agent, action_size)
        actions = np.clip(actions, -1, 1)  
        #agent action
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]                 # env step                    
        next_states = env_info.vector_observations               # get the next state        
        rewards = env_info.rewards                               # get the reward        
        #fake rewards being the sum of all rewards since they are essentially one person
        fake_rewards = [sum(rewards) for i in rewards]
        dones = env_info.local_done                              # see if episode has finished        
        agent.step(states, actions, fake_rewards, next_states, dones) # agent step
        score += rewards                                         # update the score
        states = next_states                                     # roll over the state to next time step        
        if np.any(dones):                                        # exit loop if episode finished        
            break     
            
    #book keeping
    scores.append(np.max(score))
    scores_avg.append(np.max(score))
    moving_scores.append(np.mean(scores_avg))

    #print scores intermittenly
    if episode % 100 ==0: print(f'Episode: {episode} Score: {np.max(score)} Average Score: {np.mean(scores_avg)}')

    #break if done
    if (np.mean(scores_avg) > 0.5) & (len(scores_avg) == 100):
        print(f'Environment solved in {episode} episodes! Average Score: {np.mean(scores_avg)}')
        break    
```