import numpy as np
import matplotlib.pyplot as plt

# set index 0 as open, 1 as closed(same as in the class!)

# model after measurement
measurement_model = np.array(
    [[0.9, 0.5],
     [0.1, 0.5],]
)

# model after action - for bel_bar (never change because robot always pushes)
action_model = np.array(
    [[1, 0.6],
     [0, 0.4]]
)

bel = np.array(
    [[0.5],
     [0.5]]
)

measurements = [ 0, 0, 0, 0, 0, 1, 0, 0 ]
belief_open = []

for measurement in measurements:
    bel_bar = action_model @ bel
    unnormalized_posterior = (
        measurement_model * \
        np.repeat(bel_bar, 2, axis=1).transpose())[measurement]
    posterior = unnormalized_posterior / sum(unnormalized_posterior)
    belief_open.append(round(posterior[0], 3))
    bel = np.array([ posterior ]).transpose()


plt.plot(range(1, len(belief_open)+1), belief_open, marker='o')
plt.title("Belief - Open")
plt.grid(True)
plt.show()
