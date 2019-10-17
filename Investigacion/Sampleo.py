from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

rate, data = wavfile.read('canciones\hakuna_matata.wav')
data = data[5000000:5000100] 
print(data)