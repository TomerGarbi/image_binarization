import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pre_process
from threshold import otsu, niBlack, sauvola, wolf
# ---- k-means algorithm -----

# generates k random centroids
def random_centroids(k, n,):
    return np.random.uniform(0, 1, (k, n))


# make sure that every gaussian has different mean
def check_if_exist(e, k):
    for g in k:
        if g[0] == e:
            return True
    return False


# find the new centroids for the data with respect to the new labels
def find_new_centroids(data, labels, k):
    counters = np.zeros(k)
    acc = np.zeros((k, len(data[0])))
    i = 0
    while i < len(data):
        label = labels[i]
        acc[label] += data[i]
        counters[label] += 1
        i = i + 1
    for i in range(k):
        if counters[i] != 0:
            acc[i] = acc[i] / counters[i]
    return acc


# iterates over the data until all vectors find their center
# returns the centroids, labels and number of iterations
def k_means(data, k, MAX_ITER=100, MIN_CHANGES=0):
    labels = np.zeros(len(data), int)
    centroids = random_centroids(k, len(data[0]))
    loop = True
    iter = 0
    while (loop):
        print(f"iteration: {iter}")
        changed = 0
        margins = np.array([centroids - v for v in data])
        norms = np.array([np.linalg.norm(m.T, axis=0) for m in margins])
        new_lables = np.array([np.argmin(n) for n in norms])
        N = len(data)
        for i in range(N):
            if labels[i] != new_lables[i]:
                changed += 1
        labels = new_lables
        centroids = find_new_centroids(data, labels, k)
        loop = changed > MIN_CHANGES and iter < MAX_ITER  # this ables to determine how many changes are sufficient to stop
        iter += 1
    return centroids, labels, iter


def kmeans_threshold(img, k=9, method="otsu"):
    methods = {"otsu": otsu, "niBlack": niBlack, "sauvola": sauvola, "wolf": wolf}
    pixel_list = img.reshape((img.shape[0] * img.shape[1],img.shape[2]))
    classifier = KMeans(k, random_state=0, max_iter=15)
    result = classifier.fit(pixel_list)
    centroids = np.round(result.cluster_centers_)
    labels = result.labels_
    for i in range(len(pixel_list)):
        pixel_list[i] = centroids[labels[i]]
    new_img = pixel_list.reshape(img.shape)
    g = pre_process.grayscale(img)
    B = methods[method]((g, f"{method}_cluster.jpg"))
    return B


if __name__ == "__main__":
    lenna = cv2.imread("./images/old_recipe.jpg")
    kmeans_threshold(lenna, method="niBlack")