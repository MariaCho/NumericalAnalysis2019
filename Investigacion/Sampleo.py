from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

rate, data = wavfile.read('C:\Users\usuario\OneDrive\Documents\EAFIT\Semestre9\Analisis numerico\Investigacion\NumericalAnalysis2019\Investigacion\canciones\hakuna_matata.wav')
data = data[5000000:5000100] 
print(data)