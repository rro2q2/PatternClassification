__author__ = 'stamaimer'

import argparse
import numpy as np
from pylab import *
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gen_mean(mean):

    tmp = [float(element) for element in mean]

    # print tmp

    return tmp


def gen_cov(cov):

    tmp = []

    for element in cov:

        tmp.append([float(ele) for ele in element[1:-1].split(',')])

    # print tmp

    return tmp

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("-m", "--mean", help="Mean of the N-dimensional distribution. For example: M1,M2,M3,...,Mn")

    argument_parser.add_argument("-c", "--cov", help='Covariance matrix of the distribution. For example: "[C11,C12,C13,...,C1n], [C21,C22,C23,...,C2n], [C31,C32,C33,...,C3n], ..., [Cn1,Cn2,Cn3,...,Cnn]". Note the seperator between lists!!!')

    argument_parser.add_argument("-s", "--size", type=int, help="")

    args = argument_parser.parse_args()

    mean = gen_mean(args.mean.split(','))

    cov = gen_cov(args.cov.split(", "))

    samples = np.random.multivariate_normal(mean, cov, args.size)

    x, y = samples.T

    # print samples
    #
    # samples = sorted(samples)
    #
    # pdf = stats.norm.pdf(samples, mean, cov)
    #
    # plt.plot(samples, pdf, "-o")
    #
    # plt.show()

    rx = np.linspace(np.amin(x), np.amax(x), 200)
    ry = np.linspace(np.amin(y), np.amax(y), 200)

    X, Y = np.meshgrid(rx, ry)

    Z = bivariate_normal(X, Y)

    figure = plt.figure()

    axes = figure.add_subplot(111, projection='3d')

    axes.plot_surface(X, Y, Z, cmap="binary")

    plt.show()


