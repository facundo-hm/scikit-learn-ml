import numpy as np
import matplotlib.pyplot as plt

CHARTS_PATH = './charts/'

def plot_mnist_digit(image_data: np.ndarray, image_name: str):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.savefig(CHARTS_PATH + image_name)

def save_representative_imgs(representative_imgs: np.ndarray, k:int):
    for index, X_representative_digit in enumerate(representative_imgs):
        plt.subplot(k // 10, 10, index + 1)
        plt.imshow(
            X_representative_digit.reshape(8, 8),
            cmap='binary', interpolation='bilinear')
        plt.axis('off')

    plt.savefig(CHARTS_PATH + 'representative_images_plot')
