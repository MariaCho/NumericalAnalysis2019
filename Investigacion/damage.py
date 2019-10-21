import random
import numpy as np

def noiseadd(y, percent, rate):
    """
    Returns <y> with a noisy random <percent> number of points.
    Noisiness means the value which was y0 becomes a value between [0, 2 * y0]
    * Maybe it's not a good function (it's a completely random criteria).
    * I've read that the gaussian distribution is a good way to add noise to a signal.
    It'd great to try using it instead of this crappy method.
    """
    for i in range(len(y)):
        if random.random() < percent:
            y[i] *= random.uniform(1 - rate, 1 + rate)

def zerofill(y, percent, blocksize=1):
    """Makes a <percent> of the values 0. It's a trivial function."""
    i = 0
    while i < len(y):
        if random.random() < percent:
            m = min(blocksize, len(y) - i)
            y[i:i + m] = [0] * m
            i += m
        else:
            i += 1

def invert(y, percent):
    """Makes <percent> of the values become: y -> -y. Another trivial one."""
    for i in range(len(y)):
        if random.random() < percent:
            y[i] = -y[i]

def repeat(y, percent):
    """
    <percent> of the points become: y[n] = y[n - 1].
    Why? Maybe it can work somehow.
    """
    for i in range(1, len(y)):
        if random.random() < percent:
            y[i] = y[i - 1]