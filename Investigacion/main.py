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
x, y = utils.tovalidxy(newsamples, matches)
x = np.array(x).reshape((-1, 1))
y = np.array(y)

xP1, xP2 = np.split(x,2)
yP1, yP2 = np.split(y,2)

#plt.scatter(xP1, yP1, s=10, color='b')
#plt.scatter(xP2, yP2, s=10, color='g')
#plt.show()

polynomial_features_p1= PolynomialFeatures(degree=10)
polynomial_features_p2= PolynomialFeatures(degree=10)
x_poly_p1 = polynomial_features_p1.fit_transform(xP1)
x_poly_p2 = polynomial_features_p2.fit_transform(xP2)

model_p1 = LinearRegression()
model_p2 = LinearRegression()
model_p1.fit(x_poly_p1, yP1)
model_p2.fit(x_poly_p2, yP2)
y_poly_pred_p1 = model_p1.predict(x_poly_p1)
y_poly_pred_p2 = model_p2.predict(x_poly_p2)

#rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
#r2 = r2_score(y,y_poly_pred)
#print(rmse)
#print(r2)

#plt.scatter(x, y, s=10)

# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip_p1 = sorted(zip(xP1,y_poly_pred_p1), key=sort_axis)
sorted_zip_p2 = sorted(zip(xP2,y_poly_pred_p2), key=sort_axis)
xP1, y_poly_pred_p1 = zip(*sorted_zip_p1)
xP2, y_poly_pred_p2 = zip(*sorted_zip_p2)
plt.plot(samples, label='real')
plt.scatter(xP1, yP1, s=10, color='b')
plt.scatter(xP2, yP2, s=10, color='b')
plt.plot(xP1, y_poly_pred_p1, color='m')
plt.plot(xP2, y_poly_pred_p2, color='g')
plt.show()
#-------------------------

#fcubic = interp1d(x, y, kind='cubic', fill_value='extrapolate')

#utils.repair(newsamples, matches, fcubic)

#plt.title('Cubic')
#plt.xlabel('Frame')
#plt.ylabel('Amplitude')
#plt.plot(samples, label='real')
#plt.plot(newsamples, label='interpolated')
#plt.legend(loc='best')

#plt.show()