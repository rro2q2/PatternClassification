__author__ = 'stamaimer'

from scipy import spatial
import argparse
import numpy as np


# def gen_mean(mean):
#
#     tmp = [float(element) for element in mean]
#
#     # print tmp
#
#     return tmp


def gen_cov(cov):

    tmp = []

    for element in cov:

        tmp.append([float(ele) for ele in element[1:-1].split(',')])

    # print tmp

    return np.array(tmp)


def gen_coordinate(arg):

    tmp = [int(ele) for ele in arg[1:-1].split(',')]

    # print tmp

    return np.array(tmp)

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("-x", help='For example, "(x1,x2,x3,...,xn)"')

    argument_parser.add_argument("-y", help='For example, "(y1,y2,y3,...,yn)"')

    argument_parser.add_argument("-c", "--cov", help='Covariance matrix of the distribution. For example: "[C11,C12,C13,...,C1n], [C21,C22,C23,...,C2n], [C31,C32,C33,...,C3n], ..., [Cn1,Cn2,Cn3,...,Cnn]". Note the seperator between lists!!!')

    args = argument_parser.parse_args()

    x = gen_coordinate(args.x)

    y = gen_coordinate(args.y)

    cov = gen_cov(args.cov.split(", "))

    if len(x) != len(y):

        print "The length mismatch."

    else:

        if np.linalg.inv(cov) == 0:

            print "The cov is a singular matrix."

        else:

            print spatial.distance.mahalanobis(x, y, np.linalg.inv(cov))