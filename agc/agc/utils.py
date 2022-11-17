from math import exp, log
import numpy as np
from warnings import warn

# Converting from Mel to Hertz
# Uses formulae from Slaney's Auditory Toolbox mfcc.m
def mel2hz(z):
    
    f_0 = 0
    f_sp = 200/3
    brkfrq = 1000
    brkpt = (brkfrq - f_0)/f_sp
    logstep = exp(log(6.4)/27)

    # Using np.vectorize to do quick element-wise comparisons
    def g(x):
        if x < brkpt:
            return f_0 + f_sp * x
        else:
            return brkfrq*exp(log(logstep)*(x - brkpt))
    return np.vectorize(g)(z)

# Converting from Hertz to Mel
# Uses formulae from Slaney's Auditory Toolbox mfcc.m
def hz2mel(f):
    
    f_0 = 0
    f_sp = 200/3
    brkfrq = 1000
    brkpt = (brkfrq - f_0)/f_sp
    logstep = exp(log(6.4)/27)

    # Using np.vectorize to do quick element-wise comparisons
    def g(x):
        if x < brkfrq:
            return (x - f_0) / f_sp
        else:
            return brkpt + (log(x / brkfrq))/log(logstep)
    return np.vectorize(g)(f)

# Element-wise maximum
def mat_max(m):
    return np.vectorize(lambda x: max(0, x), otypes=[float])(m)

# Normalise the output of scipy.wavfile.read to match the implementation of Matlab's audioread
def audio_normalise(data):
    dtype = data.dtype

    # dtype: [bits, signed]
    dtypes = {
        np.int16: [16, 1],
        np.int32: [32, 1],
        np.int64: [64, 1],
        np.float32: [32, 1],
        np.float64: [64, 1],
        np.uint16: [16, 0],
        np.uint32: [32, 0],
        np.uint64: [64, 0]
    }

    try:
        bits, signed = dtypes[next(dtype_k for dtype_k in dtypes.keys() if dtype == dtype_k)]
        scaling_factor = pow(2, bits-signed) # Scaling factor for Python to Matlab

        return np.divide(data, scaling_factor)
    except:
        warn('Audio normalisation failed, automatic gain control may not work as expected.')
        return data
    
# Normalise the data to match scipy.wavfile.write (from Matlab audioread-like data)
def audio_inverse_normalise(data, target_type):

    # dtype: [bits, signed]
    dtypes = {
        np.int16: [16, 1],
        np.int32: [32, 1],
        np.int64: [64, 1],
        np.float32: [32, 1],
        np.float64: [64, 1],
        np.uint16: [16, 0],
        np.uint32: [32, 0],
        np.uint64: [64, 0]
    }

    # Set default
    if target_type not in dtypes.keys(): target_type = np.int16

    try:
        bits, signed = dtypes[next(dtype_k for dtype_k in dtypes.keys() if target_type == dtype_k)]
        scaling_factor = 1 / (pow(2, bits-signed)-1) # Scaling factor for Matlab to Python

        data = data / np.max(np.abs(data))  # Normalise to +-1
        return (np.divide(data, scaling_factor)).astype(target_type)
    except:
        warn('Audio normalisation failed, automatic gain control may not work as expected.')
        return data
    

# Testing
if __name__ == '__main__':
    # Mel-Hertz Conversion
    hz = mel2hz([2])
    print(hz)
    print(hz2mel(hz))

    mel = hz2mel(8000)
    print(mel)
    print(mel2hz(mel))

    # Python-Matlab Audio Normalisation
    from scipy.io import wavfile
    
    test_audio_path = 'test/speech.wav'
    sample_rate, data = wavfile.read(test_audio_path)
    print(data[0:10])
    data = audio_normalise(data)
    data = audio_inverse_normalise(data, target_type=np.int16)
    print(data[0:10])
    wavfile.write('output.wav', sample_rate, data)