import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow.keras.layers import Input, Lambda, InputLayer, Reshape, Conv2D, Flatten, Dense, Conv2DTranspose

#preprocess
def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

if __name__=="__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #config
    configs = {
        'train_size': 60000,
        'batch_size': 32,
        'test_size': 10000,
        'latent_dim': 2,
        'epochs': 30,
        'epsilon_std': 1.0
    }

    # Data Load
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    # train, test preprocess

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(configs['train_size']).batch(configs['batch_size']))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                   .batch(configs['batch_size']))


    #############################
    #  create a smapling layer  #
    #############################
    """
    reconstruction error를 계산할 때 Monte-carlo technique로 계산하면 되지만 z가 ramdom으로 추출되기 때문에
    backpropagation이 적용이 되지 않는다. reparameterization Trick을 통해 N(0,1)에서 램던함수 epsilon를 추출 
    """
    class Sampling(tf.keras.layers.Layer):
        """
        Users (z_mean, z_log_var) to sample z, the vector encodinga digit.
        """
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    #############################
    #    build the encoder      #
    #############################

    encoder_inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(configs['latent_dim'], name='z_mean')(x)
    z_log_var = Dense(configs['latent_dim'], name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    #############################
    #    build the decoder      #
    #############################

    latent_inputs = Input(shape=(configs['latent_dim'], ))
    x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder):
            super(VAE, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        @tf.function
        def train_step(self, data):
            with tf.GradientTape() as tape:
                total_loss, reconstruction_loss, kl_loss, reconstruction = self.compute_loss(data)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

        def compute_loss(self, data):
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    # Train data - reconstruction
                )
            )
            kl_loss = 0.5 * (tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            return total_loss, reconstruction_loss, kl_loss, reconstruction

    vae = VAE(encoder, decoder)
    vae.compile(tf.keras.optimizers.Adam(1e-4))
    epochs = configs['epochs']
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            vae.train_step(train_x)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(vae.compute_loss(test_x)[0])
        total_loss = loss.result()
        print('Epoch: {}, Test set total_loss: {}, time elapse for current epoch: {}'
              .format(epoch, total_loss, end_time - start_time))

    def plot_latent_space(vae, n=30, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.savefig('vae.png')
        plt.show()

    plot_latent_space(vae)