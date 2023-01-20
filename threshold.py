import numpy as np
import matplotlib.pyplot as plt
import cv2
import pre_process


def apply_threshold(img, T):
    rows, cols = img.shape
    bw = np.zeros((rows, cols))
    if isinstance(T, int):
       bw[img >= T] = 255
    elif isinstance(img, np.ndarray):
        for i in range(rows):
            for j in range(cols):
                bw[i][j] = 0 if img[i][j] < T[i][j] else 255
    return bw

def calculate_otsu_threshold(img: np.ndarray, T: int):
    number_of_pixels = img.size
    T_img = np.zeros(img.shape)
    T_img[img >= T] = 1
    pixels1 = np.count_nonzero(T_img)
    pixels0 = number_of_pixels - pixels1
    w0 = pixels0 / number_of_pixels
    w1 = pixels1 / number_of_pixels
    if w0 == 0 or w1 == 0:
        return -1
    pixels0_indices = img[T_img == 0]
    pixels1_indices = img[T_img == 1]

    V0 = np.var(pixels0_indices)
    V1 = np.var(pixels1_indices)

    return w0 * V0 + w1 * V1
    

def otsu(img):
    criterias = []
    for th in range(255):
        c = calculate_otsu_threshold(img[0], th)
        if c != -1:
            criterias.append((c, th))
    T = min(criterias, key=lambda x: x[0])
    bw_img  = apply_threshold(img[0], T[1])
    cv2.imwrite(f"./result_images/otsu/{img[1]}", bw_img)



def get_window(img, r, c, W, mode="move"):
    rows, cols = img.shape
    left = c - W // 2 - 1
    right = c + W // 2
    ceil = r - W // 2 - 1
    bottom = r + W // 2

    # adjust window to valid indices
    if mode == "move":
        if left < 0:
            right += abs(left)
            left = 0
        elif right >= cols:
            left -= right + 1 - cols
            right = cols - 1

        if ceil < 0:
            bottom += abs(ceil)
            ceil = 0
        elif bottom >= rows:
            ceil -= bottom + 1 - rows
            bottom = rows - 1
        return img[ceil: bottom, left:right]
    
    elif mode == "pad":
        if left < 0:
            left = 0
        elif right >= img.shape[0]:
            right = img.shape[0] - 1
        if ceil < 0:
            ceil = 0
        elif bottom >= img.shape[1]:
            bottom = img.shape[1] - 1
        window= img[ceil: bottom, left:right]
        pad_window = window
        return pad_window
        


def window_mean(window):
    return np.mean(window)

def window_var(window, mu):
    return np.var(window)


def mean_std(img, W):
    rows, cols = img.shape
    M = np.zeros(img.shape)
    S = np.zeros(img.shape)
    for r in range(rows):
        for c in range(cols):
            print(r * c / (rows * cols))
            pixel_window = get_window(img, r, c, W)    
            M[r][c] = window_mean(pixel_window)
            S[r][c] = np.sqrt(window_var(pixel_window, M[r][c]))
    return M, S


def niBlack(img, k=-0.2, W=15):
    M, S = mean_std(img, W)
    T = M + k * S
    bw_img = apply_threshold(img[0], T)
    cv2.imwrite(f"./result_images/niBlack/{img[1]}", bw_img)
    return bw_img

def sauvola(img, k=0.5, W=15, R=128):
    M, S = mean_std(img, W)
    rows, cols = img.shape
    bw_img = np.zeros((rows, cols))
    for r in range(rows):
        print(f"{r / rows}%")
        for c in range(cols):
            T = M[r][c] * (1 + k * (S[r][c]/R - 1))
            bw_img[r][c] = 0 if img[r][c] <= T else 255
    return bw_img


def wolf(img, k=0.5, W=15, R=128):
    M, S = mean_std(img, W)
    max_S = np.max(S)
    min_M = np.min(M)
    rows, cols = img.shape
    bw_img = np.zeros((rows, cols))
    for r in range(rows):
        print(f"{r / rows}%")
        for c in range(cols):
            T = M[r][c] - k * (1 - (S[r][c]/max_S)) * (M[r][c] - min_M)
            bw_img[r][c] = 0 if img[r][c] <= T else 255
    return bw_img



def salt_and_pepper(img):
    pre_process.add_noise(img)



if __name__ == "__main__":
    mode = 'w'
    lenna = pre_process.grayscale(cv2.imread("./images/old_recipe.jpg"), mode=mode)
    A = sauvola(lenna , W=9)
    cv2.imwrite("test.jpg", A)