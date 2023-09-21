import math
import matplotlib.pyplot as plt
import random


if __name__ == "__main__":
    observations_generated = []
    for _ in range(100000):
        observations_generated.append(math.exp(-1*random.uniform(0, 1)))
    
    observations = [random.expovariate(1.0) for _ in range(100000)]

    plt.hist(observations_generated)
    plt.show()
    plt.hist(observations)
    plt.show()