import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_most_common_color(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_temp = img.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    img_temp[:, :, 0], img_temp[:, :, 1], img_temp[:, :, 2] = unique[np.argmax(counts)]
    return img_temp

# get the most similar color tone that is generally the dominant color in road signs
def get_color_name(img):

    h, w, c = img.shape

    red = [255, 0, 0]
    blue = [0, 0, 255]
    white = [255, 255, 255]
    black = [0, 0, 0]

    colors = {'black':black, 'red':red, 'blue':blue, 'white':white}
    min_dist = 999999
    pixel_color = img[h//2, w//2]
    color = None
    for k, v in colors.items():
        dist = np.linalg.norm(v-pixel_color)
        if dist < min_dist:
            min_dist = dist
            color = k
    return color


def show_dominant_color(img_1, img_2):
    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off')
    ax[1].axis('off')
    f.tight_layout()


