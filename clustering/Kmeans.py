import time
import sys
import numpy as np
import matplotlib.pyplot as plt

import math
import os
from matplotlib.pyplot import imread
# from mpl_toolkits.mplot3d import Axes3D #This is for 3d scatter plots.

import cv2
import functools
from sklearn.decomposition import PCA
from random import sample
import random

random.seed(1)


def initialise_parameters(m, X):
    C = sample(list(X), k=m)
    return np.array(C)


def distance(centre, sample):
    return (math.sqrt(np.sum((centre - sample) ** 2)))


#     return(math.sqrt((centre[0]-sample[0])**2+(centre[1]-sample[1])**2))
def E_step(C, X):
    L = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        ags = np.argmin(np.linalg.norm(C - X[i], axis=1))
        L[i] = C[ags]
    return L


def M_step(C, X, L):
    new_C = np.zeros(C.shape)
    for i in range(0, C.shape[0]):
        centroid = C[i]
        count = np.count_nonzero(np.all(L == centroid, axis=1))
        new_C[i] = np.sum(X[np.all(L == C[i], axis=1)], axis=0) / count
    return new_C


def kmeans(X, m, threshold):
    L = np.zeros(X.shape)
    C = initialise_parameters(m, X)
    diff = float('inf')
    loss_prev = float('inf')
    while diff > threshold:
        L = E_step(C, X)
        loss_cur = np.sum((np.linalg.norm(X - L, axis=1) ** 2))
        diff = abs(loss_prev - loss_cur)
        loss_prev = loss_cur
        C = M_step(C, X, L)
    # due to the fact that we pair the data point and cluster centroid
    # by the value in L match the value in C, we need to do E_step one more time
    L = E_step(C, X)
    return C, L


# C_final, L_final = kmeans(X, k, 1e-6)

# print(C_final)
def allocator(X, L, c):
    cluster = []
    for i in range(L.shape[0]):
        if np.array_equal(L[i, :], c):
            cluster.append(X[i, :])
    return np.asarray(cluster)


def evaluation(X, L_final, C_final, k):
    length = 0
    total_distance = 0  # evaluation
    density = 0
    for i in range(k):
        cluster = allocator(X, L_final, C_final[i, :])
        #         print("the "+str(i)+" cluster is: "+str(cluster[:,i]))
        length += len(cluster)
        for d in range(cluster.shape[0]):
            total_distance += distance(C_final[i, :], cluster[d, :])
        density += total_distance / len(cluster)
    density = total_distance / k
    return length, total_distance, density


# X = np.load("./data.npy")
def main():
    train_images = []
    path = "../output"
    for file in os.listdir(path):
        if file.endswith(".png"):
            im = cv2.imread(path + "/" + file)
            y = list(im.ravel())
            y = np.array(y)
            # im =imread(path + "/" + file)
            train_images.append(y)

    X = np.array(train_images)
    m, n = X.shape
    pca = PCA(n_components=min(m, n), random_state=2023)
    pca.fit(X)

    X_pca = pca.transform(X)

    X = X_pca

    xaxis = []
    densityaxis = []
    yaxis = []
    wrongtimes = 0
    for i in range(1, 50):
        xaxis.append(i)
        k = i
        C_final, L_final = kmeans(X, k, 1e-6)
        length, total_distance, density = evaluation(X, L_final, C_final, k)
        yaxis.append(total_distance)
        densityaxis.append(density)
        print(k, end=" ")
        if (length != X.shape[0]):
            print("Error, when i = " + str(i) + " the clusters start overlapping: " + str(length))
            wrongtimes += 1
        else:
            wrongtimes = 0

        if (wrongtimes >= 5):
            print("error wrong times >= 5")
            break

    fig = plt.figure(figsize=(16, 10))
    plt.plot(xaxis, densityaxis)
    plt.xlabel("Number of potential cluster")
    plt.ylabel("sparseness")
    plt.show()


if __name__ == '__main__':
    main()
