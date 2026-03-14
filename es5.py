""" Utilizzare un frammento audio complesso per studiare come la riduzione dei livelli di
ampiezza influenzi la qualità del segnale (trovate nella cartella «materiale» alcuni
esempi di audio).
1. Importare un file audio.
2. Estrarre un singolo canale (mono) e normalizzare l'ampiezza nell'intervallo [−1,1].
3. Utilizzare la funzione di quantizzazione implementata nell'esercizio 4 per
ridurre la risoluzione del segnale originale a 8, 4 e 2 bit.
Test d'Ascolto e Confronto:
1. Riprodurre le versioni a risoluzione ridotta. Descrivere come cambia la
percezione del suono.
2. Riprodurre il segnale rumore a 2 bit. Cosa si sente? """

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# we extract the audio into an array (data)
fs, data = wavfile.read('Materiale/gong.wav')

# if the data is stereo, we take the average of the two channels
if data.ndim == 2:
    data = data.mean(axis=1)

data = data.astype(np.float64)
data = data / np.max(np.abs(data)) # easy normalization step: we make everything positive, find the max value of the data array, and devide each component with that maximum

def quantizza(segnale, n_bit, A_min=-1, A_max=1):
    livelli = 2 ** n_bit
    delta = (A_max - A_min) / (livelli - 1)  # we also can do without the -1
    y_quant = np.round(segnale / delta) * delta
    return y_quant

data8bit = quantizza(data, 8)
data4bit = quantizza(data, 4)
data2bit = quantizza(data, 2)

sd.play(data, samplerate=fs)
sd.wait()

sd.play(data8bit, samplerate=fs)
sd.wait()

sd.play(data4bit, samplerate=fs)
sd.wait()

sd.play(data2bit, samplerate=fs)
sd.wait()