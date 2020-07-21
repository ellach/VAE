import os
import matplotlib.pyplot as plt
import cv2

''' Inspired by from https://towardsdatascience.com/image-pre-processing-c1aec0be3edf '''


def load_image(path):
    image_file = sorted([os.path.join(path, 'train', file)
                         for file in os.listdir(path + "/train")
                         if file.endswith('.png')])
    return image_file


def display(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()


def display1(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()


def processing(data):
    img = [cv2.normalize(cv2.imread(i, cv2.IMREAD_UNCHANGED), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for i in data[:1]]
    height = 220
    width = 220
    dim = (height, width)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        no_noise.append(blur)

    return no_noise


path = ' '
data = load_image(path)
DATASET = processing(data)
