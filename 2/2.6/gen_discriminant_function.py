__author__ = 'stamaimer'

import numpy as np
import argparse


def gen_mean(mean):

    tmp = [float(element) for element in mean]

    # print tmp

    return np.array(tmp)


def gen_cov(cov):

    tmp = []

    for element in cov:

        tmp.append([float(ele) for ele in element[1:-1].split(',')])

    # print tmp

    return np.array(tmp)


def gen_discriminant_function_of_normal_distribution(mean, cov, prior):

    d = len(mean)

    if isinstance(mean, np.ndarray) \
            and isinstance(cov, np.ndarray) \
            and np.linalg.det(cov) != 0 \
            and cov.shape == (d, d) \
            and prior >= 0 \
            and prior <= 1:

        def discriminant_function_of_normal_distribution(x):

            if isinstance(x, np.ndarray) and len(x) == d:

                return -0.5 * np.dot(np.dot((x - mean).transpose(), np.linalg.inv(cov)), (x - mean)) - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)

            else:

                print "x is not a ndarray or the length of x is mismatch"

                return None

        return discriminant_function_of_normal_distribution

    else:

        print "mean isn't a ndarray or cov isn't a ndarray or cov is a singular matrix or the shape of cov is mismatch or the prior is invalidate"

        return None


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("-m", "--mean", help="Mean of the N-dimensional distribution. For example: M1,M2,M3,...,Mn")

    argument_parser.add_argument("-c", "--cov", help='Covariance matrix of the distribution. For example: "[C11,C12,C13,...,C1n], [C21,C22,C23,...,C2n], [C31,C32,C33,...,C3n], ..., [Cn1,Cn2,Cn3,...,Cnn]". Note the seperator between lists!!!')

    argument_parser.add_argument("-p", "--prior", type=float, help="The prior probability")

    args = argument_parser.parse_args()

    mean = gen_mean(args.mean.split(','))

    cov = gen_cov(args.cov.split(", "))

    discriminant_function_of_normal_distribution = gen_discriminant_function_of_normal_distribution(mean, cov, args.prior)

    print discriminant_function_of_normal_distribution