import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

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


if __name__ == "__main__":
    start_time = time.time()
    lenna = cv2.imread("./images/old_newspaper.jpg")
    lenna2 = cv2.imread("./images/Lenna.png")
    g = grayscale(lenna)
    cv2.imwrite("ppp.png", g)
    print("running time: ", time.time() - start_time)
    cv2.imshow("original", lenna)
    cv2.waitKey(0)
    cv2.destroyAllWindows()