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



def get_window(shape, r, c, W):
    left = c - W // 2 - 1
    right = c + W // 2
    ceil = r - W // 2 - 1
    bottom = r + W // 2
    # adjust window to valid indices
    if left < 0:
        right += abs(left)
        left = 0
    elif right >= shape[0]:
        left -= right + 1 -shape[0]
        right = shape[0] - 1
    
    if ceil < 0:
        bottom += abs(ceil)
        ceil = 0
    elif bottom >= shape[1]:
        ceil -= bottom + 1 - shape[1]
        bottom = shape[1] - 1
    return (left, right, bottom, ceil)


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
            (left, right, bottom, ceil) = get_window((rows, cols), r, c, W)
            pixel_window = img[ceil:bottom, left:right] 
            M[r][c] = window_mean(pixel_window)
            S[r][c] = np.sqrt(window_var(pixel_window, M[r][c]))
    return M, S


def niBlack(img, k=-0.2, W=15):
    M, S = mean_std(img[0], W)
    T = M + k * S
    bw_img = apply_threshold(img[0], T)
    cv2.imwrite(f"./result_images/niBlack/{img[1]}", bw_img)
    return bw_img

def sauvola(img, k=0.5, W=15, R=128):
    rows, cols = img[0].shape
    bw_img = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            (left, right, bottom, ceil) = get_window((rows, cols), r, c, W)
            pixel_window = img[0][ceil:bottom, left:right] 
            pixel_mu = window_mean(pixel_window)
            pixel_std = np.sqrt(window_var(pixel_window, pixel_mu))
            T = pixel_mu*(1+ k*(pixel_std/R - 1))
            bw_img[r][c] = 0 if img[0][r][c] <= T else 255
    cv2.imwrite(f"./result_images/sauvola/{img[1]}", bw_img)
    return bw_img


def wolf(img, k=0.5, W=15, R=128):
    M, S = mean_std(img[0], W)
    max_S = np.max(S)
    min_M = np.min(M)
    rows, cols = img[0].shape
    bw_img = np.zeros((rows, cols))
    for r in range(rows):
        print(f"{r / rows}%")
        for c in range(cols):
            T = M[r][c] - k * (1 - (S[r][c]/max_S)) * (M[r][c] - min_M)
            bw_img[r][c] = 0 if img[0][r][c] <= T else 255
    cv2.imwrite(f"./result_images/Wolf/{img[1]}", bw_img)
    return bw_img
    


if __name__ == "__main__":
    mode = 'w'
    lenna = pre_process.grayscale(cv2.imread("./images/Lenna.png"), mode=mode)
    print("1")
    newspaper = pre_process.grayscale(cv2.imread("./images/old_newspaper.jpg"), mode=mode)
    print("2")
    stop_sign = pre_process.grayscale(cv2.imread("./images/stop_sign.jpg"), mode=mode)
    print('3')
    sudoku1 = pre_process.grayscale(cv2.imread("./images/sudoku.jpg"), mode=mode)
    print('4')
    sudoku2 = pre_process.grayscale(cv2.imread("./images/sudoku2.jpg"), mode=mode)
    gray_images = [(lenna, "lenna.png"), (newspaper, "newspaper.jpg"), (stop_sign, "stop_sign.jpg"),
                 (sudoku1, "sudoku1.jpg"), (sudoku2, "sudoku2.jpg")]
    for img in gray_images:
        print(img[1])
        # otsu(img)
        # niBlack(img)
        # sauvola(img)
        wolf(img)



