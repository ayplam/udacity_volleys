import json
from collections import deque
import numpy as np

## To keep track of experiments
def append_to_file(text, filename):
    with open(filename, "a") as f:
        f.write(text + "\n")
        
def write_json(dd, filename):
    with open(filename, "w") as f:
        json.dump(dd, f, indent=4)
        

def check_env_solved(env, agent, num_episodes = 100):

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    agent.reset()
    scores_deque = deque(maxlen=100)
    
    for _ in range(num_episodes):
        score = np.zeros(num_agents)
        while True:
            # Keep acting until the env is done
            states = env_info.vector_observations
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            score += env_info.rewards
            dones = env_info.local_done
            if np.any(dones):
                break

        scores_deque.append(np.mean(score))
        
    return scores_deque