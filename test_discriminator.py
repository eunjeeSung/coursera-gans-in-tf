# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from discriminator import Discriminator

# %%
def test_discriminator(z_dim, hidden_dim, num_test=100):
    tf.random.set_seed(2)

    disc = Discriminator(z_dim, hidden_dim).get_disc()

    # Check there are three parts
    assert len(disc.layers) == 4

    # Check the linear layer is correct
    test_input = tf.random.normal([num_test, z_dim], 0, 1)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)

test_discriminator(5, 10)
test_discriminator(20, 8)
print("Success!")
# %%
