__author__ = 'stamaimer'

import seaborn
import argparse
import collections
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


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

    print samples.tolist()

    # print collections.Counter(samples.tolist())

    samples = sorted(samples)

    pdf = stats.norm.pdf(samples, mean, cov)

    plt.plot(samples, pdf, "-o")

    plt.show()


