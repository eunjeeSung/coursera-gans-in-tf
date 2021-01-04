# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%
def gen_block(output_dim):
    return keras.Sequential(
        [
            layers.Dense(output_dim, activation=None),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
    )

# %%
def disc_block(output_dim):
    return keras.Sequential(
        [
            layers.Dense(output_dim, activation=None),
            layers.LeakyReLU()
        ]
    )
# %%
