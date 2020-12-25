import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %%
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = tf.stop_gradient(image_tensor)
    image_unflat = tf.reshape(image_tensor, (-1, *size))
    image_grid = make_grid(image_unflat[:num_images], grid_row=5)
    #plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    #plt.imshow(tf.transpose(image_grid, perm=(1, 2, 0)))
    plt.imshow(image_grid)
    plt.show()

# %%
def make_grid(images, grid_row):
    '''
    logic from https://github.com/tensorflow/tensorflow/blob/r1.8/
        tensorflow/contrib/gan/python/eval/python/eval_utils_impl.py
    '''
    grid_col = images.shape[0] // grid_row
    img_h, img_w = images[0].shape[1], images[0].shape[2]
    height, width = grid_row * img_h, grid_col * img_w

    grid = tf.reshape(images, (grid_row, grid_col, img_h, img_w))
    grid = tf.transpose(grid, perm=(0, 2, 1, 3))
    grid = tf.reshape(grid, (height, width))

    return grid

# %%
def get_noise(n_samples, z_dim):
    return tf.random.normal([n_samples, z_dim])

