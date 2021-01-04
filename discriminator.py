# %%
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from blocks import disc_block

# %%
class Discriminator(keras.Model):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = keras.Sequential(
            [
                disc_block(hidden_dim * 4),
                disc_block(hidden_dim * 2),
                disc_block(hidden_dim),
                layers.Dense(1)
        ]
        )

    def call(self, image):
        return self.disc(image)
    
    def get_disc(self):
        return self.disc
# %%
