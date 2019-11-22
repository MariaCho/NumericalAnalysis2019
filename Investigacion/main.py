#! /usr/bin/env python

import operator
import damage, recognize, utils
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

samplerate, samples = wavfile.read('canciones/hakuna_matata.wav')
samples = samples[5000000:5000100]

newsamples = samples.copy()
damage.noiseadd(newsamples, 0.7, 0.3)
matches = recognize.cheat(samples, newsamples, false_positives=0.04, false_negatives=0.1)
matchesSD = recognize.cheat(samples, samples, false_positives=0.04, false_negatives=0.1)
x, y = utils.tovalidxy(newsamples, matches)
xSD, ySD = utils.tovalidxy(samples, matchesSD)
x = np.array(x).reshape((-1, 1))
y = np.array(y)
xSD = np.array(xSD).reshape((-1, 1))
ySD = np.array(ySD)

xP, yP, xSD, ySD = utils.partir(x, y, xSD, ySD, 10)
polynomial_features = PolynomialFeatures(degree=10)
y_poly_pred = utils.polinomialR(xP, yP, ySD, polynomial_features)
utils.plotting(xP, yP, samples, y_poly_pred)
