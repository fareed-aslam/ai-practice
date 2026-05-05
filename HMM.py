import numpy as np

states = ["Sunny", "Cloudy", "Rainy"]
observations = ["Umbrella", "NoUmbrella"]

start_prob = [0.5, 0.3, 0.2]

transition = [
    [0.6, 0.3, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
]

emission = [
    [0.1, 0.9],
    [0.4, 0.6],
    [0.8, 0.2]
]

def simulate_hmm(days):
    state = np.random.choice(states, p=start_prob)
    seq_states = []
    seq_obs = []

    for _ in range(days):
        state_index = states.index(state)
        obs = np.random.choice(observations, p=emission[state_index])

        seq_states.append(state)
        seq_obs.append(obs)

        state = np.random.choice(states, p=transition[state_index])

    return seq_states, seq_obs

hidden, obs = simulate_hmm(10)

print("Hidden States:", hidden)
print("Observations:", obs)
