import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# same as ex.5
fs, data = wavfile.read('Materiale/gong.wav')

if data.ndim == 2:
    data = data.mean(axis=1)

data = data.astype(np.float64)
data = data / np.max(np.abs(data)) # easy normalization step: we make everything positive, find the max value of the data array, and devide each component with that maximum

def quantizza(segnale, n_bit, A_min=-1, A_max=1):
    livelli = 2 ** n_bit
    delta = (A_max - A_min) / (livelli - 1)
    segnale_clipped = np.clip(segnale, A_min, A_max)
    y_quant = np.round((segnale_clipped - A_min) / delta) * delta + A_min
    return y_quant

data4bit = quantizza(data, 4)
levels = 2 ** 4
delta = (1 - (-1)) / (levels - 1)

# let's create an empty array
noise = np.zeros(len(data))
# let's put random values with the same amplitude as the quantization step
for i in range(len(data)):
    noise[i] = np.random.uniform(-delta/2, delta/2)

data_dithered = data + noise
data_dithered4bit = quantizza(data_dithered, 4)

# MSE
errore_4bit = data - data4bit
mse_4bit = np.mean(errore_4bit ** 2)

errore_4bitnoise = data - data_dithered4bit
mse_4bitnoise = np.mean(errore_4bitnoise ** 2)

print(f'Standard deviation between the original audio and the 4 bit quantization without noise: {mse_4bit}')
print(f'Standard deviation between the original audio and the 4 bit quantization with added noise: {mse_4bitnoise}')

# noise - signal rateo in dB
snr_4bit = 10 * np.log10(np.mean(data**2) / np.mean(errore_4bit**2))
snr_dithered = 10 * np.log10(np.mean(data**2) / np.mean(errore_4bitnoise**2))

print(f'4 bit SNR with noise: {snr_dithered:.2f} dB')
print(f'4 bit SNR: {snr_4bit:.2f} dB')

sd.play(data, samplerate=fs)
sd.wait()
sd.play(data4bit, samplerate=fs)
sd.wait()
sd.play(data_dithered4bit, samplerate=fs)
sd.wait()

# Without dither, the quantization error is correlated to the signal, meanwhile if we add noise, it's more random and less dependent on the signal.
# For the same reason, we see higher results in the standard deviation between original data and dithered data
# This is also why the dithered audio seems less metallic than the one we only quantized