import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Lambda, Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

LAMBDA = 1.13
BETTA = 0.3


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# network parameters
input_shape = (original_dim,)
intermediate_dim = 512
batch_size = 128
latent_dim = 128
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
numbers = Input(shape=(10,), name='number_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
concated_layer = Concatenate()([z, numbers])
# instantiate encoder model
encoder = Model([inputs, numbers], [z_mean, z_log_var, concated_layer], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
# build decoder model
latent_inputs = Input(shape=(latent_dim + 10,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
# instantiate VAE model
outputs = decoder(encoder([inputs, numbers])[2])
vae = Model([inputs, numbers], outputs, name='vae_mlp')
if __name__ == '__main__':
    models = (encoder, decoder)
    data = (x_test, y_test)


    def vae_loss(y_true, y_pred):
        z_mean = vae.get_layer('encoder').get_layer('z_mean').output
        z_log_var = vae.get_layer('encoder').get_layer('z_log_var').output
        z_var = tf.math.exp(z_log_var)
        z_mean_square = tf.math.square(z_mean)
        z_var_square = tf.math.square(z_var)

        reconstruction_loss = binary_crossentropy(y_true, y_pred)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - z_mean_square - z_var
        kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        z_var_square_inv = tf.math.reciprocal(z_var_square)

        first_term = tf.matmul(z_var_square, tf.transpose(z_var_square_inv))

        r = tf.matmul(z_mean * z_mean, tf.transpose(z_var_square_inv))

        r2 = z_mean * z_mean * z_var_square_inv
        r2 = tf.reduce_sum(r2, 1)

        second_term = 2 * tf.matmul(z_mean, tf.transpose(z_mean * z_var_square_inv))
        second_term = r - second_term + tf.transpose(r2)

        r = tf.reduce_sum(tf.math.log(z_var_square), 1)
        r = tf.reshape(r, [-1, 1])
        third_term = r - tf.transpose(r)

        PPP = 0.5 * (first_term + second_term + third_term - 128)

        return + (1 + LAMBDA) * BETTA * reconstruction_loss \
               + tf.math.reduce_mean(kl_loss) \
               + LAMBDA * tf.math.reduce_mean(PPP)


    vae.compile(optimizer='adam', loss=vae_loss, experimental_run_tf_function=False)
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    # train the autoencoder
    y_train_hot = np.zeros((y_train.shape[0], 10))
    y_train_hot[np.arange(y_train.shape[0]), y_train] = 1

    y_test_hot = np.zeros((y_test.shape[0], 10))
    y_test_hot[np.arange(y_test.shape[0]), y_test] = 1

    vae.fit([x_train, y_train_hot], x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([x_test, y_test_hot], x_test)
            )

    encoder.save('tmp/encoder.h5')
    decoder.save('tmp/decoder.h5')
