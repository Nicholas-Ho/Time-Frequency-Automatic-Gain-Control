# Short-Term Fourier Transform (and Inverse) as implemented by Ellis

from math import floor
import numpy as np
from numpy.fft import fft, ifft

def stft(x, fft_length, window_size, hop_length):
    window = np.hanning(window_size) if window_size != 0 else np.ones(fft_length)

    X = np.zeros((1+fft_length//2, 1+floor((len(x)-fft_length)//hop_length)), dtype=np.complex_)

    i = 0
    for j in range(0, len(x)-fft_length, hop_length):
        u = np.multiply(window, x[j:j+fft_length])
        t = fft(u)
        X[:, [i]] = np.transpose(np.asmatrix(t[0:1+fft_length//2]))
        i += 1

    return X # Data is complex

def istft(X, fft_length=None, window_size=0, hop_length=None):
    # Set defaults
    if fft_length == None: fft_length = 2*(len(X)-1)

    if len(X) != fft_length/2 + 1:
        raise Exception('Number of rows should be fft_length/2+1')

    window = np.hanning(window_size)*2/3 if window_size != 0 else np.ones(fft_length)

    if hop_length == None: hop_length = len(window)//2

    x = np.zeros(fft_length + (X.shape[1]-1) * hop_length)

    for i in range(0, (X.shape[1]-1) * hop_length, hop_length):
        ft = X[:, i//hop_length]
        ft = np.squeeze(np.asarray(ft))
        ft = np.concatenate((ft, np.conjugate(ft[(fft_length//2-1):0:-1])))
        px = np.real(ifft(ft))
        x[i:i+fft_length] = x[i:i+fft_length] + np.multiply(px, window)

    return x