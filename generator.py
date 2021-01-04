# %%
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from blocks import gen_block

# %%
class Generator(tf.keras.Model):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = keras.Sequential(
            [
                gen_block(hidden_dim),
                gen_block(hidden_dim * 2),
                gen_block(hidden_dim * 4),
                gen_block(hidden_dim * 8),
                layers.Dense(im_dim),
                layers.Activation('sigmoid')
            ]
        )

    def call(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen

# %%
