# Generate a matrix of weights to combine FFT bins into Mel bins

from math import ceil
from .utils import mel2hz, hz2mel, mat_max
import numpy as np

def fft2mel_mx(nfft, sample_rate, nfilts, width=1, minfrq=0, maxfrq=None, constamp=0):
    '''
    nfilts specifies output bands required
    width is the constant width of each band relative to standard Mel
    '''
    if maxfrq == None: maxfrq = sample_rate / 2

    # Just in case
    if nfilts == 0:
        nfilts = ceil(hz2mel(maxfrq)/2)

    weights = np.zeros((nfilts, nfft))

    # Generate center frequencies of each FFT bin
    fft_frqs = (np.linspace(0, 0.5, nfft//2+1) * sample_rate)

    # Generate center 'frequencies' of Mel bands (uniformly spaced between limits)
    minmel = hz2mel([minfrq])[0]
    maxmel = hz2mel([maxfrq])[0]
    bin_frqs = mel2hz(minmel + np.linspace(0, 1, nfilts+2)*(maxmel-minmel))

    for i in range(0, nfilts):
        fs = bin_frqs[i:i+3]
        # Scale by width
        fs = fs[1] + width * (fs - fs[1])
        # Calculate Lower and Upper slopes for all bins
        loslope = (fft_frqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fft_frqs) / (fs[2] - fs[1])
        # Then intersect them with each other and zero
        weights[i, 0:nfft//2+1] = mat_max(np.minimum(loslope, hislope))

    if constamp == 0:
        # Slaney-style Mel is scaled to be approximately constant E per channel
        weights = np.matmul(np.diag(2/(bin_frqs[2:nfilts+2] - bin_frqs[0:nfilts])), weights)

    weights[:, nfft//2+2:nfft] = 0

    return weights, bin_frqs

# Testing
if __name__ == '__main__':
    weights, bin_frqs = fft2mel_mx(512, 16000, 20, 2)
    print(weights)