import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist


def evaluate():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

    # load json and create model
    encoder = load_model('./tmp/encoder.h5', compile=False)
    decoder = load_model('./tmp/decoder.h5', compile=False)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 10  # how many digits we will display
    fix, ax = plt.subplots(10, 11, figsize=(17, 18), dpi=100)
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

    plt.savefig("./test_results/" + str(int(time.time())) + ".png")
    plt.show()

    exit()
