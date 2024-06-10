import numpy as np
import matplotlib.pyplot as plt

def plot_mnist_digit(image_data: np.ndarray):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
