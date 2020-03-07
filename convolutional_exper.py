import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def visualize_data(img):
    plt.gray()
    plt.imshow(img)
    plt.show()

def apply_convolution(weight, filter, size_x, size_y):
    for x in range(1,size_x-1):
        for y in range(1,size_y-1):
            convolution = 0.0
            convolution = convolution + (img[x - 1, y - 1] * filter[0][0])
            convolution = convolution + (img[x - 1, y] * filter[0][1])
            convolution = convolution + (img[x - 1, y + 1] * filter[0][2])
            convolution = convolution + (img[x, y - 1] * filter[1][0])
            convolution = convolution + (img[x, y] * filter[1][1])
            convolution = convolution + (img[x, y + 1] * filter[1][2])
            convolution = convolution + (img[x + 1, y - 1] * filter[2][0])
            convolution = convolution + (img[x + 1, y] * filter[2][1])
            convolution = convolution + (img[x + 1, y + 1] * filter[2][2])
            convolution = convolution * weight
            if(convolution<0):
                convolution=0
            if(convolution>255):
                convolution=255
            img_transformed[x, y] = convolution

    return img_transformed

def apply_pooling(img_transformed, size_x, size_y):
    new_size_x = int(size_x/2)
    new_size_y = int(size_y/2)
    pooling_image = np.zeros((new_size_x, new_size_y))
    for x in range(0, size_x, 2): # after two pixels
        for y in range(0, size_y, 2):
            pixels = []
            pixels.append(img_transformed[x, y])
            pixels.append(img_transformed[x, y + 1])
            pixels.append(img_transformed[x + 1, y])
            pixels.append(img_transformed[x + 1, y + 1])
            pooling_image[int(x/2), int(y/2)] = max(pixels)

    return pooling_image

if __name__ == "__main__":

    img = misc.ascent()
    # visualize_data(img)

    img_transformed = np.copy(img)
    size_x = img_transformed.shape[0]
    size_y = img_transformed.shape[1]

    print(size_x, size_y)

    #Experiment with different values for fun effects.
    filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    # filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    weight = 1
    img_transformed = apply_convolution(weight, filter, size_x, size_y)
    visualize_data(img_transformed)

    pooling_image = apply_pooling(img_transformed, size_x, size_y)
    visualize_data(pooling_image)