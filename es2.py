import numpy as np
import matplotlib.pyplot as plt

# --- Parametri ---
f_nota = 440 # LA4
f_camp = 44100
durata = 2

# --- Vettori temporali ---
fs_alta = 10000
t = np.arange(0, durata, 1/f_camp) # misura da 0 a 2 secondi, con 10k misurazioni al secondo. tot misurazioni = 20k
x = np.sin(2 * np.pi * f_nota * t)

# --- Riproduzione audio ---
import sounddevice as sd

sd.play(x, samplerate=44100)
sd.wait()

sd.play(x, samplerate=88200)
sd.wait()

sd.play(x, samplerate=22050)
sd.wait()