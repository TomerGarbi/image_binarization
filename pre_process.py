import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.signal import wiener, deconvolve
from scipy.ndimage import median_filter
import threshold

def gray_avg(pixel):
    return sum(pixel) // 3

def gray_max(pixel):
    return max(pixel)

def gray_min(pixel):
    return min(pixel)

def gray_weights(pixel):
    return int(0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2])


def grayscale(img: np.ndarray, mode="avg"):
    gray_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.int16)
    modes = {
        "avg": gray_avg,
        "max": gray_max,
        "min": gray_min,
        'w': gray_weights
    }
    f = modes[mode]
    for i in range(len(gray_img)):
        for j in range(len(gray_img[i])):
            gray_img[i][j] = f(img[i][j])
    #gray_img = np.apply_along_axis(f, 2, img)
    
    return gray_img


def histogram(img: np.ndarray):
    c = 0
    hist = np.zeros((256))
    for r in img:
        for p in r:
            if p == 150:
                c += 1
            hist[p] += 1
    return hist



def add_gaussian_noise(img):
    row,col,ch= img.shape
    mean = 0
    var = 30
    gauss = np.random.normal(mean,var,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    gauss = np.round(gauss)
    print(gauss)
    noisy = img + gauss
    return noisy


def add_noise(img):
    rows, cols = img.shape
    number_of_pixels = np.random.randint(300, rows * cols // 10)
    for i in range(number_of_pixels):
        y_coord = np.random.randint(0, rows - 1)
        x_coord = np.random.randint(0, cols - 1)
        img[y_coord][x_coord] = 255
     
    number_of_pixels = np.random.randint(300 , rows * cols // 10)
    for i in range(number_of_pixels):
        y_coord = np.random.randint(0, rows - 1)
        x_coord = np.random.randint(0, cols - 1)
        img[y_coord][x_coord] = 0      
    return img

if __name__ == "__main__":
    start_time = time.time()

    print(sapper.shape)
    median_res = cv2.medianBlur(sapper, 3)
    