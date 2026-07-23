import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import sounddevice as sd


# --- Parameters ---
f_nota = 440 # LA4
f_camp = 44100
durata = 2
f_noise = 50
t = np.arange(0, durata, 1/f_camp)

# --- Signals ---
x_pulito = np.sin(2 * np.pi * f_nota * t)
ronzio = 0.5 * np.sin(2 * np.pi * f_noise * t)
fruscio = 0.2 * np.random.randn(len(t))

x_disturbato = x_pulito + ronzio + fruscio

# --- Filters ---

# Passa-alto: taglia sotto 150 Hz
b_alto, a_alto = butter(2, 150 / (f_camp/2), btype='high')
# Passa-basso: taglia sopra 3000 Hz
b_basso, a_basso = butter(2, 3000 / (f_camp/2), btype='low')

# Applica i filtri in cascata
x_filtrato = filtfilt(b_alto, a_alto, x_disturbato)  # prima passa-alto
x_filtrato = filtfilt(b_basso, a_basso, x_filtrato)  # poi passa-basso

# --- Visualizzazione ---
t_zoom = t[:2000]  # zoom sui primi ~45ms per leggere la forma d'onda

fig, assi = plt.subplots(3, 2, figsize=(14, 8))
fig.suptitle("Esercizio 1 - Pulizia segnale audio")

segnali = [x_pulito, x_disturbato, x_filtrato]
titoli  = ["Originale (440 Hz)", "Disturbato (+50Hz +rumore)", "Filtrato"]

for i, (segnale, titolo) in enumerate(zip(segnali, titoli)):

    # Dominio del tempo (zoom)
    assi[i, 0].plot(t_zoom, segnale[:2000])
    assi[i, 0].set_title(f"{titolo} - Tempo")
    assi[i, 0].set_xlabel("Tempo [s]")
    assi[i, 0].set_ylabel("Ampiezza")

    # Dominio delle frequenze (FFT)
    N = len(segnale)
    fft_vals = np.abs(np.fft.rfft(segnale)) / N
    fft_freq = np.fft.rfftfreq(N, 1/f_camp)

    assi[i, 1].plot(fft_freq, fft_vals)
    assi[i, 1].set_title(f"{titolo} - Frequenze")
    assi[i, 1].set_xlabel("Frequenza [Hz]")
    assi[i, 1].set_ylabel("Ampiezza")
    assi[i, 1].set_xlim(0, 1000)  # zoom sulla banda di interesse

plt.tight_layout()
plt.show()

# --- Audio ---
def normalizza(s):
    return (s / np.max(np.abs(s))).astype(np.float32)

print("Originale..."); sd.play(normalizza(x_pulito), f_camp); sd.wait()
print("Disturbato..."); sd.play(normalizza(x_disturbato), f_camp); sd.wait()
print("Filtrato...");   sd.play(normalizza(x_filtrato), f_camp); sd.wait()

# note: we can't see the noise (line 17) because it has an amplitude of 0.2 distributed on many (random) frequencies, not like the 50Hz noise or the 440Hz sinwave