# %%
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from utils import show_tensor_images

# %%
def main():
    # TODO: Clean up
    ds = tfds.load('mnist', split='train', as_supervised=True)
    ds = ds.take(25)

    test_images = []
    for image, label in tfds.as_numpy(ds):
        test_images.append(image.flatten())
    test_images = tf.convert_to_tensor(np.array(test_images))

    show_tensor_images(test_images, num_images=50)
#main()


# %%
if __name__ == 'main':
    main()
