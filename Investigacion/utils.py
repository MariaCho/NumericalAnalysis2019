import numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def tovalidxy(y, marks):
    """
    Returns two arrays: x and y, where only points marked as True
    are considered.
    """
    #n = marks.count(True)
    n = len(marks)

    validx = [None] * n
    validy = [None] * n

    cnt = 0
    #for i in range(len(y)):
    #    if marks[i]:
    #        validx[cnt] = i
    #        validy[cnt] = y[i]
    #        cnt += 1
    for i in range(len(y)):
        validx[cnt] = i
        validy[cnt] = y[i]
        cnt += 1
        
    return validx, validy

def repair(y, marks, fn):
    """
    Replaces the value of the <y> if the value of <marks> is
    false in that index. y[i] = fn[i]
    """
    for i in range(len(y)):
        if not marks[i]:
            y[i] = fn(i)

def invalidx(matches):
    """
    Array of indexes of matches[i] == False.
    """
    return [i for i in range(len(matches)) if not matches[i]]

def replace(y, indexes, newvalues):
    """
    Replaces values at indexes[i] with newvalues[i] in y
    """
    for index, newval in zip(indexes, newvalues):
        y[index] = newval

def partir(x, y, xSD, ySD, numero):

    xP = [None]*numero
    yP = [None]*numero

    xP = np.split(x,numero)
    yP = np.split(y,numero)

    xSDAux = [None]*numero
    ySDAux = [None]*numero

    xSDAux = np.split(xSD,numero)
    ySDAux = np.split(ySD,numero)

    return xP, yP, xSDAux, ySDAux

def polinomialR(xP, yP, ySD, polynomial_features):
    x_poly = [None] * len(xP)
    model = [None] * len(xP)
    y_poly_pred = [None] * len(xP)
    
    for i in range(len(model)): 
        model[i] = LinearRegression()

    for i in range(len(x_poly)):
        x_poly[i] = polynomial_features.fit_transform(xP[i])
        model[i].fit(x_poly[i], yP[i])
        y_poly_pred[i] = model[i].predict(x_poly[i])
        rmse = np.sqrt(mean_squared_error(ySD[i],y_poly_pred[i]))
        r2 = r2_score(ySD[i],y_poly_pred[i])
        print("RMSE: ",rmse,"R2: ",r2)
    
    return y_poly_pred

def sortedAxis(xP, y_poly_pred):
    sort_axis = operator.itemgetter(0)
    sorted_zip = [None] * len(xP)

    for i in range(len(sorted_zip)):
        sorted_zip[i] = sorted(zip(xP[i],y_poly_pred[i]), key=sort_axis)
        xP[i], y_poly_pred[i] = zip(*sorted_zip[i])
    
    return xP, y_poly_pred

def plotting(xP, yP, samples, y_poly_pred):

    plt.plot(samples, label='real', color='k')

    for i in range(len(xP)):
        color = 'C' + str(i)
        plt.scatter(xP[i], yP[i], s=10, color='k')
        plt.plot(xP[i], y_poly_pred[i], color=color )
        plt.legend(loc='best')
    
    plt.show()
