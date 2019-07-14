import numpy as np


def generate_HMM_observation(num_obs, pi, T, E):
    def drawFrom(probs):
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    obs = np.zeros(num_obs)
    states = np.zeros(num_obs)
    states[0] = drawFrom(pi)
    obs[0] = drawFrom(E[:, int(states[0])])
    for t in range(1, num_obs):
        states[t] = drawFrom(T[int(states[t-1]), :])
        obs[t] = drawFrom(E[:, int(states[t])])
    return np.int64(obs), states
