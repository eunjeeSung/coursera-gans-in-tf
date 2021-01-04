# %%
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers

from generator import Generator
from discriminator import Discriminator

from utils import get_noise, show_tensor_images

# %%
criterion = losses.BinaryCrossentropy(from_logits=True)
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

gen_ckpt_path = 'checkpoints/gen/gen-{epoch:04d}.ckpt'
disc_ckpt_path = 'checkpoints/disc/disc-{epoch:04d}.ckpt'
gen_ckpt_dir = os.path.dirname(gen_ckpt_path)
disc_ckpt_dir = os.path.dirname(disc_ckpt_path)
save_epoch_step = 50

# %%
ds_train = tfds.load('mnist',
                split='train',
                shuffle_files=True,
                batch_size=batch_size,
                as_supervised=True)                  

# %%
gen = Generator(z_dim)
gen_opt = optimizers.Adam(learning_rate=lr)
disc = Discriminator()
disc_opt = optimizers.Adam(learning_rate=lr)

# %%
def train():
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    for epoch in range(1, n_epochs+1):    
        # Load data
        for real, _ in tfds.as_numpy(ds_train):
            cur_batch_size = len(real)

            # Flatten images
            real = real.astype("float32") / 255.0
            real = tf.reshape(real, (cur_batch_size, -1))

            # Take step
            d_loss, g_loss = train_step(gen, disc, gen_opt, disc_opt, criterion, real, cur_batch_size, z_dim)

            # Keep track of the average losses
            mean_discriminator_loss += d_loss.numpy() / display_step
            mean_generator_loss += g_loss.numpy() / display_step

            # Visualization
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, \
                        discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim)
                fake = gen(fake_noise, training=False)
                show_tensor_images(fake)
                #show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1

        # Save model
        if epoch % save_epoch_step == 0:
            gen.save_weights(gen_ckpt_path.format(epoch=epoch))
            disc.save_weights(disc_ckpt_path.format(epoch=epoch))
#train()

# %%
@tf.function
def train_step(gen, disc, gen_opt, disc_opt, criterion, real, cur_batch_size, z_dim):
    # Discriminator
    with tf.GradientTape() as disc_tape:
        d_loss = disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim)
    disc_grads = disc_tape.gradient(d_loss, disc.trainable_weights)
    disc_opt.apply_gradients(zip(disc_grads, disc.trainable_weights))

    # Generator
    with tf.GradientTape() as gen_tape:
        g_loss = gen_loss(gen, disc, criterion, cur_batch_size, z_dim)
    gen_grads = gen_tape.gradient(g_loss, gen.trainable_weights)
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_weights))

    return d_loss, g_loss

# %%
def disc_loss(gen, disc, criterion, real, num_images, z_dim):
    # 1) Create noise and generate fake images
    noise_tensor = get_noise(num_images, z_dim)
    fake = gen(noise_tensor, training=False)

    # 2) Get prediction of fake image & Calculate Loss
    fake_predict = disc(fake, training=True)
    fake_target = tf.zeros([num_images, 1])
    fake_loss = criterion(fake_target, fake_predict)

    # 3) Get prediction of real image & Calculate Loss
    real_predict = disc(real, training=True)
    real_target = tf.ones([num_images, 1])
    real_loss = criterion(real_target, real_predict)

    # 4) Average real * fake loss
    disc_loss = (fake_loss + real_loss) / 2

    return disc_loss

# %%
def gen_loss(gen, dsic, criterion, num_images, z_dim):
    # 1) Create noise and generate fake images    
    noise_tensor = get_noise(num_images, z_dim)
    fake = gen(noise_tensor, training=True)

    # 2) Get prediction of fake image & Calculate Loss    
    disc_predict = disc(fake, training=True)
    target = tf.ones([num_images, 1])
    gen_loss = criterion(target, disc_predict)

    return gen_loss

