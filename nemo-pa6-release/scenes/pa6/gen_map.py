# ----------------------------------------------------------------------------
# Copyright (c) 2026. Columbia University. All rights reserved.
#
# This software and documentation contain confidential and proprietary
# information that is the property of Columbia University.
#
# Unauthorized copying, distribution, or modification of this file,
# via any medium, is strictly prohibited.
#
# Project code of COMS W4167 by Changxi Zheng (cxz@cs.columbia.edu)
# ----------------------------------------------------------------------------

def gen1():
    for i in range(200):
        for j in range(256):
            if i < 100:
                print('.', end="")
            elif j < 128:
                print('o', end="")
            else:
                print('.', end="")
        print()

def gen2():
    for i in range(100):
        for j in range(100):
            if i < 50:
                print('.', end="")
            elif j < 70 and j > 40:
                print('o', end="")
            else:
                print('.', end="")
        print()
if __name__ == '__main__':
    gen2()
    #gen1()
