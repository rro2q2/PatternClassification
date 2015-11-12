__author__ = 'stamaimer'


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mean = [0, 0]
cov = [[1, 0], [0, 100]]

x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)

X, Y = np.meshgrid(x, y)

Z = np.random.multivariate_normal(mean, cov, 10)

figure = plt.figure()

axes = figure.add_subplot(111, projection='3d')

axes.plot_surface(X, Y, Z)



