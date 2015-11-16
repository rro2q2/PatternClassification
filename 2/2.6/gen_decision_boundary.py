__author__ = 'stamaimer'


import numpy as np

mean1 = np.array([ 1, 0])

mean2 = np.array([-1, 0])

cov = np.array([[1, 0], [0, 1]])

p1 = p2 = 0.5

X0 = 0.5 * (mean1 + mean2) - (np.log(p1 / p2) / np.dot(np.dot((mean1 - mean2).transpose(), np.linalg.inv(cov)), (mean1 - mean2))) * (mean1 - mean2)

W = np.dot(np.linalg.inv(cov), (mean1 - mean2))

