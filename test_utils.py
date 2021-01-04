# %%
import tensorflow as tf

from utils import get_noise

# %%
def test_get_noise(n_samples, z_dim):
    noise = get_noise(n_samples, z_dim)
    
    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (n_samples, z_dim)
    assert tf.abs(noise.numpy().std() - tf.constant(1.0)) < 0.01

test_get_noise(1000, 100)
print("Success!")

# %%
