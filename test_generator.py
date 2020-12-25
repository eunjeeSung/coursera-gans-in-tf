# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from generator import Generator

# %%
def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
    tf.random.set_seed(2)
    
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
    
    # Check there are six modules in the sequential part
    assert len(gen.layers) == 6
    test_input = tf.random.normal([num_test, z_dim], 0, 1)
    test_output = gen(test_input)

    # Check that the output shape is correct
    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.numpy().max() < 1, "Make sure to use a sigmoid"
    assert test_output.numpy().min() > 0, "Make sure to use a sigmoid"
    print(test_output.numpy().std())
    assert test_output.numpy().std() > 0.05, "Don't use batchnorm here"
    assert test_output.numpy().std() < 0.15, "Don't use batchnorm here"

# %%
test_generator(5, 10, 20)
test_generator(20, 8, 24)
print("Success!")

