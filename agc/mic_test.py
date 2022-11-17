import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from agc.agc import AutomaticGainController

fs = 48000  # Sample rate
seconds = 5  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished


filename = 'output.wav'

write(filename, fs, myrecording)  # Save as WAV file 

# Extract data and sampling rate from file
data, fs = sf.read(filename, dtype='float32')  
sd.play(data, fs)
status = sd.wait()  # Wait until file is done playing


# Testing AGC processing
controller = AutomaticGainController(t_scale=0.5, f_scale=5)
myrecording = controller.process_audio(myrecording, fs, smooth_gain=True, noise_reduce=True)

write(filename, fs, myrecording)  # Save as WAV file 

# Extract data and sampling rate from file
data, fs = sf.read(filename, dtype='float32')  
sd.play(data, fs)
status = sd.wait()  # Wait until file is done playing