import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Lambda, Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
import test

LAMBDA = 1.13
BETTA = 1


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon


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
x = Dense(intermediate_dim, activation='relu')(x)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
concated_layer = Concatenate()([z, numbers])
# instantiate encoder model
encoder = Model([inputs, numbers], [z_mean, z_log_var, concated_layer], name='encoder')
encoder.summary()
plot_model(encoder, to_file='tmp/vae_mlp_encoder.png', show_shapes=True)
# build decoder model
latent_inputs = Input(shape=(latent_dim + 10,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
x = Dense(intermediate_dim, activation='relu')(x)
x = Dense(intermediate_dim, activation='relu')(x)
outputs = Dense(original_dim, activation='sigmoid')(x)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='tmp/vae_mlp_decoder.png', show_shapes=True)
# instantiate VAE model
outputs = decoder(encoder([inputs, numbers])[2])
vae = Model([inputs, numbers], outputs, name='vae_mlp')
if __name__ == '__main__':
    models = (encoder, decoder)
    data = (x_test, y_test)


    def reconstruction_loss(y_true, y_pred):
        loss = original_dim * binary_crossentropy(y_true, y_pred)

        return (1 + LAMBDA) * BETTA * loss


    def kl_loss1(y_true, y_pred):
        mean = vae.get_layer('encoder').get_layer('z_mean').output
        log_var = vae.get_layer('encoder').get_layer('z_log_var').output
        var = tf.math.exp(log_var)
        mean_square = tf.math.square(mean)

        # kl_loss  q(z|x)||p(z)  q(z|x)||N(0,1)
        loss = 1 + z_log_var - mean_square - var
        loss = tf.math.reduce_sum(loss, axis=-1)
        loss *= -0.5

        return loss


    def kl_loss2(y_true, y_pred):
        mean = vae.get_layer('encoder').get_layer('z_mean').output
        log_var = vae.get_layer('encoder').get_layer('z_log_var').output
        var = tf.math.exp(log_var)

        var_square = tf.math.square(var)
        var_square_reciprocal = tf.math.reciprocal(var_square)

        first_term = tf.matmul(var_square, tf.transpose(var_square_reciprocal))

        r = tf.matmul(mean * mean, tf.transpose(var_square_reciprocal))

        r2 = mean * mean * var_square_reciprocal
        r2 = tf.reduce_sum(r2, 1)

        second_term = 2 * tf.matmul(mean, tf.transpose(mean * var_square_reciprocal))
        second_term = r - second_term + tf.transpose(r2)

        r = tf.reduce_sum(tf.math.log(var_square), 1)
        r = tf.reshape(r, [-1, 1])
        third_term = r - tf.transpose(r)

        loss = 0.5 * tf.math.reduce_mean(first_term + second_term + third_term - latent_dim)

        return LAMBDA * loss


    def vae_loss(y_true, y_pred):
        return reconstruction_loss(y_true, y_pred) \
               + kl_loss1(y_true, y_pred) \
               + kl_loss2(y_true, y_pred)


    vae.compile(optimizer='adam',
                loss=vae_loss,
                experimental_run_tf_function=False,
                metrics=[reconstruction_loss, kl_loss1, kl_loss2])

    vae.summary()
    plot_model(vae,
               to_file='tmp/vae_mlp.png',
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


test.evaluate()
