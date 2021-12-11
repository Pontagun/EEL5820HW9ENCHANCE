import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt

greyscale = 256

def find_lowerb(r):
    for i in range(len(r)):
        if r[i] > 0:
            return i


def find_upperb(r):
    for i in range(len(r) - 1, 0, -1):
        if r[i] > 0:
            return i


def hist_table(img):
    img_flatten = img.flatten()

    img_row = img.shape[0]
    img_col = img.shape[1]

    N = img_row * img_col
    histogram_table_dim = (greyscale, 4)

    histogram_table = np.zeros(histogram_table_dim)

    for i, val in enumerate(img_flatten):
        histogram_table[int(val)][1] = histogram_table[int(val)][1] + 1  # nk

    trk = 0
    for t_row in range(greyscale):
        histogram_table[t_row][0] = t_row / (greyscale - 1)  # rk
        histogram_table[t_row][2] = histogram_table[t_row][1] / N  # prk
        trk = trk + histogram_table[t_row][2]
        histogram_table[t_row][3] = trk

    return histogram_table, img_flatten


def dynamic_range(image):
    hist_value = np.zeros(greyscale)
    ri = 0
    rk = 255

    fltn_image = image.flatten()
    for i, val in enumerate(fltn_image):
        hist_value[int(val)] = hist_value[int(val)] + 1

    a = find_lowerb(hist_value)
    b = find_upperb(hist_value)

    ba_diff = (b - a)
    for i, r in enumerate(fltn_image):
        rp = int(((rk - ri) * (r - a) / ba_diff) + ri)
        fltn_image[i] = rp

    return fltn_image


def mapper(hist_table):

    pixel_mapper = np.zeros(greyscale)

    print(len(hist_table[:, 0])) # sk
    print(len(hist_table[:, 3])) # tr

    tmp_tr = hist_table[:, 3]
    tmp_sk = hist_table[:, 0]
    tmp_tr = [0.04, 0.08, 0.12, 0.51, 0.53, 0.57, 0.61, 1.00]
    tmp_sk = [0.00, 0.14, 0.28, 0.43, 0.57, 0.71, 0.86, 1.00]

    for i, tr in enumerate(tmp_tr):
        for j, sk in enumerate(tmp_sk):
            if abs(tr-sk) == 0:
                pixel_mapper[i] = j
                break
            elif j == 255:
                pixel_mapper[i] = j
            elif abs(tr-tmp_sk[j]) < abs(tr-tmp_sk[j+1]):
                pixel_mapper[i] = j
                break

    return pixel_mapper


if __name__ == '__main__':
    path1 = r'after6.jpg'
    img = cv2.imread(path1, 0)

    plt.hist(img.flatten(), bins=255)
    plt.show()

    m = img.shape[0]
    n = img.shape[1]

    hist_table, img_flatten = hist_table(img)

    tb_mapper = mapper(hist_table)

    img = img.flatten()

    for i, val in enumerate(img):
        img[i] = tb_mapper[val]

    # new_img = dynamic_range(img)

    new_img = img
    new_img = new_img.reshape(m, n)

    im = Image.fromarray(new_img)
    im.save("after6.jpg")
