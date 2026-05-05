import numpy as np

states = ["Sunny", "Cloudy", "Rainy"]

transition = {
    "Sunny":  [0.6, 0.3, 0.1],
    "Cloudy": [0.3, 0.4, 0.3],
    "Rainy":  [0.2, 0.3, 0.5]
}

def simulate(days):
    current = "Sunny"
    seq = [current]

    for _ in range(days):
        current = np.random.choice(states, p=transition[current])
        seq.append(current)

    return seq

result = simulate(10)
print(result)

count = 0
trials = 10000

for _ in range(trials):
    seq = simulate(10)
    if seq.count("Rainy") >= 3:
        count += 1

print(count / trials)
