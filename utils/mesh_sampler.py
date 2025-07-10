import numpy as np

def sample_random_points(height, width, num_points=64, seed=None):
    rng = np.random.RandomState(seed)
    pts = np.stack([
        rng.randint(0, height, size=num_points),
        rng.randint(0, width, size=num_points)
    ], axis=1)
    return pts
