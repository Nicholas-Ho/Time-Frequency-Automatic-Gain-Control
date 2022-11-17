# Based on D. Ellis (2010), "Time-frequency automatic gain control", web resource, available: http://labrosa.ee.columbia.edu/matlab/tf_agc/ .

from math import log2
from functools import reduce
import numpy as np
import noisereduce as nr
import os, sys

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# Handling of relative import
if __name__ == '__main__':
    sys.path.append(DIR_PATH)

    from stft import stft, istft
    from fft2mel_mx import fft2mel_mx
    from utils import audio_normalise, audio_inverse_normalise
else:
    from .stft import stft, istft
    from .fft2mel_mx import fft2mel_mx
    from .test.agc_test import test_agc
    from .utils import audio_normalise, audio_inverse_normalise



class AutomaticGainController:

    fft2mel_mx_args = (-1, -1, -1, -1) # Impossible args
    f2a = None

    def __init__(
        self,
        # Gain control parameters
        t_scale=0.5,
        f_scale=5.0, # Default is 1.0 in reference
        normalise=False,
        # Noise reduction parameters
        noise_reduction_ratio=0.7):

        self.t_scale = t_scale
        self.f_scale = f_scale
        self.normalise = normalise
        self.noise_reduction_ratio = noise_reduction_ratio

    # Lazy getter for fft2mel_mx arguments
    def _get_f2a(self, fft_length, sample_rate, nbands, mwidth):
        
        weights_generated = self.f2a != None
        if type(weights_generated) != bool: # Then it must be an array
            weights_generated = weights_generated.any()

        same_args = self.fft2mel_mx_args != (fft_length, sample_rate, nbands, mwidth)

        if weights_generated or same_args:
            # If fft2mel weights have not been calculated before and the arguments are the different, calculate weights
            # Else, used cached weights
            self.f2a, _ = fft2mel_mx(fft_length, sample_rate, nbands, mwidth)
            self.fft2mel_mx_args = (fft_length, sample_rate, nbands, mwidth)
        return self.f2a

    # Automatic gain control: make louds sounds softer and soft sounds louder
    def _smooth_gain(self, indata, sample_rate, t_scale=0.5, f_scale=1.0, smooth_type=0, normalise=False):

        # Required for sounddevice.InputStream. Also just good to have.
        indata = indata.flatten()

        # Normalise (required for scipy.io.wavfile.read)
        if normalise:
            try:
                dtype = indata.dtype
            except:
                dtype = type(indata)
            indata = audio_normalise(indata)

        # Generate STFT on a ~32 ms grid (taking into account sample rate)
        samples = 0.032 * sample_rate
        fft_length = int(pow(2, log2(samples))) # The 2^round is to ensure that f is a power of 2
        window_size = fft_length
        hop_length = fft_length // 2

        # Short-Term Fourier Transform
        D = stft(indata, fft_length, window_size, hop_length)

        ft_sample_rate = sample_rate / hop_length

        # Convert and smooth in frequency on ~ mel resolution
        # Width of mel filters depends on how many you ask for, so ask for fewer for larger f_scales
        # Broader f_scales minimise frequency-dependent gain variation
        nbands = int(max(10, 20/f_scale)) # 20/f_scale, with a minimum of 10 (for f_scale > 2)
        mwidth = f_scale * nbands / 10 # f_scale, with a minimum of 2.0

        # Lazy Getter
        f2a_weights = self._get_f2a(fft_length, sample_rate, nbands, mwidth)

        f2a_weights = f2a_weights[:, 0:fft_length//2+1]
        audgram = np.matmul(f2a_weights, np.absolute(D))


        if smooth_type == 1:
            # Noncausal, time-symmetric smoothing
            # Smooth in time with tapered window of duration ~t_scale
            pass # Implement later
        else:
            # Traditional attack/decay smoothing
            nrow, ncol = audgram.shape
            fbg = np.zeros((nrow, ncol))
            state = np.zeros(nrow)
            alpha = np.exp(-(1/ft_sample_rate)/t_scale)

            for i in range(ncol):
                state = np.amax([alpha*state, audgram[:, i]], axis=0)
                fbg[:, i] = state

        sum_f2a = np.sum(f2a_weights, axis=0)
        # Changing the zeroes to ones
        sum_f2a[sum_f2a==0] = 1
        E = reduce(np.matmul, [np.diag(1/sum_f2a), np.asmatrix(f2a_weights).H, fbg]) # np_matrix.H returns the complex conjugate transpose

        if normalise:
            return audio_inverse_normalise(istft(np.divide(D, E)), dtype)
        else:
            return istft(np.divide(D, E))

    # Smooth gain and reduce noise
    def process_audio(self, data, sample_rate, smooth_gain=True, noise_reduce=True):
        if smooth_gain:
            data = self._smooth_gain(data, sample_rate, self.t_scale, self.f_scale, normalise=self.normalise)
        if noise_reduce:
            data = nr.reduce_noise(data, sample_rate, prop_decrease=self.noise_reduction_ratio)
        return data


# Testing
if __name__ == '__main__':

    controller = AutomaticGainController()

    # Sanity check with arbitrary data
    # controller.smooth_gain(range(8000), 16000)

    # AGC Test
    from test.agc_test import test_agc
    test_agc(controller._smooth_gain)