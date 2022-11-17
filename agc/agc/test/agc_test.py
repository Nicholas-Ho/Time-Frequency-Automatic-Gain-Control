from scipy.io import wavfile
from typing import Callable
import os

test_file = 'mic.wav'

def test_agc(agc: Callable, output_file='output.wav'):
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    sample_rate, data = wavfile.read(test_path)
    dtype = data.dtype
    data = agc(data, sample_rate, normalise=True)

    wavfile.write(output_file, sample_rate, data.astype(dtype))

if __name__ == '__main__':
    # Sanity check, should output the input
    test_agc(lambda x, y: x)
