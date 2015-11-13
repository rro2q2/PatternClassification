__author__ = 'stamaimer'


import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import gen_discriminant_function as gdf
from mpl_toolkits.mplot3d import Axes3D

data1 = [[-5.01, -8.12, -3.68],
         [-5.43, -3.48, -3.54],
         [ 1.08, -5.52,  1.66],
         [ 0.86, -3.78, -4.11],
         [-2.67,  0.63,  7.39],
         [ 4.94,  3.29,  2.08],
         [-2.51,  2.09, -2.59],
         [-2.25, -2.13, -6.94],
         [ 5.56,  2.86, -2.26],
         [ 1.03, -3.33,  4.33]]

data1 = np.array(data1)

data2 = [[-0.91, -0.18, -0.05],
         [ 1.30, -2.06, -3.53],
         [-7.75, -4.54, -0.95],
         [-5.47,  0.50,  3.92],
         [ 6.14,  5.72, -4.85],
         [ 3.60,  1.26,  4.36],
         [ 5.37, -4.63, -3.65],
         [ 7.18,  1.46, -6.66],
         [-7.39,  1.17,  6.30],
         [-7.50, -6.32, -0.31]]

data2 = np.array(data2)

data3 = [[ 5.35,  2.26,  8.13],
         [ 5.12,  3.22, -2.66],
         [-1.34, -5.31, -9.87],
         [ 4.48,  3.42,  5.19],
         [ 7.11,  2.39,  9.21],
         [ 7.17,  4.33, -0.98],
         [ 5.75,  3.97,  6.65],
         [ 0.77,  0.27,  2.41],
         [ 0.90, -0.43, -8.71],
         [ 3.52, -0.36,  6.43]]

data3 = np.array(data3)


def plt_1d(class1, class2):

    prior1 = 0.5
    prior2 = 0.5

    mean1 = np.array([np.mean(class1[:, 0])])
    mean2 = np.array([np.mean(class2[:, 0])])

    # print mean1, mean2

    cov1 = np.array([[np.cov([class1[:, 0]])]])
    cov2 = np.array([[np.cov([class2[:, 0]])]])

    # print cov1, cov2

    discriminant_function1 = gdf.gen_discriminant_function_of_normal_distribution(mean1, cov1, prior1)
    discriminant_function2 = gdf.gen_discriminant_function_of_normal_distribution(mean2, cov2, prior2)

    # X = np.linspace(np.amin(class1[:, 0]), np.amax(class1[:, 0]), 200)

    X = np.linspace(-100, 100, 100)

    y1 = [discriminant_function1(np.array([x])) for x in X]

    y2 = [discriminant_function2(np.array([x])) for x in X]

    plt.plot(X, y1)

    plt.plot(X, y2)

    plt.show()


def plt_2d(class1, class2):

    prior1 = 0.5
    prior2 = 0.5

    mean1 = np.mean(class1[:, 0:2], axis=0)
    mean2 = np.mean(class2[:, 0:2], axis=0)

    # print mean1, mean2

    cov1 = np.cov([class1[:, 0], class1[:, 1]])
    cov2 = np.cov([class2[:, 0], class2[:, 1]])

    # print cov1, cov2

    discriminant_function1 = gdf.gen_discriminant_function_of_normal_distribution(mean1, cov1, prior1)
    discriminant_function2 = gdf.gen_discriminant_function_of_normal_distribution(mean2, cov2, prior2)

    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)

    X, Y = np.meshgrid(x, y)

    z1 = [discriminant_function1(np.array([x, y])) for x, y in zip(X[0], Y[:, 0])]

    z2 = [discriminant_function2(np.array([x, y])) for x, y in zip(X[0], Y[:, 0])]

    figure = plt.figure()

    axes = figure.add_subplot(111, projection='3d')

    axes.plot_surface(X, Y, z1, cmap="Greys")

    axes.plot_surface(X, Y, z2, cmap="Blues")

    plt.show()

plt_1d(data1, data2)


