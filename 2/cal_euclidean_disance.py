__author__ = 'stamaimer'

import numpy as np
import argparse

def gen_coordinate(arg):

    tmp = [int(ele) for ele in arg[1:-1].split(',')]

    # print tmp

    return np.array(tmp)

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("-x", help='For example, "(x1,x2,x3,...,xn)"')

    argument_parser.add_argument("-y", help='For example, "(y1,y2,y3,...,yn)"')

    args = argument_parser.parse_args()

    x = gen_coordinate(args.x)

    y = gen_coordinate(args.y)

    if len(x) != len(y):

        print "The length mismatch."

    else:

        print np.linalg.norm(x - y)

