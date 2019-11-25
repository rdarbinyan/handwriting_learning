import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class Sampling(Layer):
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] //= 2
        return tuple(shape)

    def call(self, input):
        z_mean, z_log_var = input[:, :input.shape[1] // 2], input[:, input.shape[1] // 2:]
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))


# load json and create model
encoder = load_model('tmp/encoder.h5', custom_objects={'Sampling': Sampling})
decoder = load_model('tmp/decoder.h5', custom_objects={'Sampling': Sampling})

img = image.load_img('test_img.jpg', target_size=(28, 28), color_mode="grayscale")

print(np.array(img).flatten().shape)
print(x_test[66].shape)



# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
fix, ax = plt.subplots(10,11, figsize=(17,18), dpi=100)
for i in range(n):
    index = np.random.randint(0, 10000)
    x_single_test = np.array(x_test[index]).flatten() / 256
    ax[i][0].imshow(x_test[index].reshape(28, 28), cmap="gray_r")
    ax[i][0].set_xticks([])
    ax[i][0].set_yticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[i][0].spines[axis].set_linewidth(3)

    for j in range(10):
        encoded_img = encoder.predict([np.array([x_single_test, ]), np.array([np.eye(10)[j]], )])

        decoded_img = decoder.predict(encoded_img[2])

        ax[i][j + 1].imshow(decoded_img[0].reshape(28, 28), cmap="gray_r")
        ax[i][j + 1].set_xticks([])
        ax[i][j + 1].set_yticks([])
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_single_test.reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_img[0].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
plt.savefig("test.png")
plt.show()

exit()