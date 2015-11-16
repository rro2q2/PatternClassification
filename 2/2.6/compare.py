__author__ = 'stamaimer'

from gen_random_samples import gen_md_normal_plot
from matplotlib import pyplot as plt
import gen_discriminant_function as gdf
import numpy as np

prior1 = prior2 = 0.5

mean1 = np.array([ 1, 0])

mean2 = np.array([-1, 0])

cov1 = cov2 = np.array([[1, 0], [0, 1]])


def bhattacharyya(prior1, prior2, mean1, mean2, cov1, cov2):

    k = 0.125 * np.dot(np.dot((mean2 - mean1).transpose(), np.linalg.inv(0.5 * (cov1 + cov2))), mean2 - mean1) \
        + 0.5 * np.log(np.linalg.det(0.5 * (cov1 + cov2)) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))

    p = np.sqrt(prior1 * prior2) * np.exp(-k)

    return p


def classifier2(x):

    discriminant_function1 = gdf.gen_discriminant_function_of_normal_distribution(mean1, cov1, prior1)
    discriminant_function2 = gdf.gen_discriminant_function_of_normal_distribution(mean2, cov2, prior2)

    if discriminant_function1(x) > discriminant_function2(x):

        # print x, "class 1"

        return 1

    elif discriminant_function1(x) < discriminant_function2(x):

        # print x, "class 2"

        return 2
    else:

        # print x, "unsure"

        return 0


def cal_empiricals():

    empiricals = []

    for total in xrange(100, 1100, 100):

        num = total / 2

        samples1 = gen_md_normal_plot(mean1, cov1, num)
        samples2 = gen_md_normal_plot(mean2, cov2, num)

        count = 0

        for instance in samples1:

            if classifier2(instance) != 1:

                count += 1

        for instance in samples2:

            if classifier2(instance) != 2:

                count += 1

        empirical = count / float(total)

        empiricals.append(empirical)

    print empiricals

    return empiricals


if __name__ == "__main__":

    for i in xrange(10):

        bhattachar = bhattacharyya(prior1, prior2, mean1, mean2, cov1, cov2)

        empiricals = cal_empiricals()

        bhattachas = [bhattachar for _ in xrange(len(empiricals))]

        print bhattachas

        figure = plt.figure()

        plt.plot(empiricals)

        plt.plot(bhattachas)

        plt.savefig(str(i) + ".png")