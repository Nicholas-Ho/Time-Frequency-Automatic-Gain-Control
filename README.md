# Time-Frequency Automatic Gain Control
A Python port of the Matlab Time-Frequency Automatic Gain Control algorithm written by Dan Ellis (source: https://www.ee.columbia.edu/~dpwe/resources/matlab/tf_agc/).

This code implements automatic gain control for audio signals, which attempts to apply a smoothly-varying gain to an audio waveform in order to keep its energy close to some target level. This version also tries to balance the energy in individual frequency bands. It works by taking the short-time Fourier transform of the signal, smoothing that energy using time and frequency scales specified as arguments, then dividing out that scaled energy. (Dan Ellis)

The port is functionally identical to the original Matlab equivalent, although some additional functionality has been introduced.

## Differences
When a signal has been passed into the gain control, the output can be quite noisy, especially for the parts of the input signal which are silent. Hence, the included controller also denoises the output of the automatic gain control using the `noisereduce` library.

## Usage
Below is a simple example for the use of the controller:

```
from scipy.io import wavfile
from agc import AutomaticGainController

test_path = 'input.wav'
sample_rate, data = wavfile.read(test_path)
dtype = data.dtype

# Due to the processing of scipy.io.wavfile.read(), the input must be normalised.
# For any other use case, normalise=False.
controller = AutomaticGainController()
data = controller.process_audio(data, sample_rate, smooth_gain=True, noise_reduce=True, normalise=True)

output_file = 'output.wav'
wavfile.write(output_file, sample_rate, data.astype(dtype))
```

For use with the microphone, see agc/mic_test.py.

## Limitations
This algorithm does not work well for audio streams as it requires the full audio signal to process. Chunk processing might result in choppy gain control.