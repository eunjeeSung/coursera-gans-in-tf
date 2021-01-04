# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from blocks import gen_block, disc_block

# %%
def test_gen_block(in_features, out_features, num_test=1000):
    tf.random.set_seed(3)

    block = gen_block(out_features)

    # Check the three parts
    assert len(block.layers) == 3
    assert type(block.layers[0]) == layers.Dense
    assert type(block.layers[1]) == layers.BatchNormalization
    assert type(block.layers[2]) == layers.ReLU
    
    # Check the output shape
    test_input = tf.random.normal([num_test, in_features], 0, 1)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)

test_gen_block(25, 12)
test_gen_block(15, 28)
print("Success!")

# %%
def test_disc_block(in_features, out_features, num_test=10000):
    tf.random.set_seed(3)

    block = disc_block(out_features)

    # Check there are two parts
    assert len(block.layers) == 2
    test_input = tf.random.normal([num_test, in_features], 0, 1)
    test_output = block(test_input)

    # Check that the shape is right
    assert tuple(test_output.shape) == (num_test, out_features)

test_disc_block(25, 12)
test_disc_block(15, 28)
print("Shape Success")
# %%
