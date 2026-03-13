import numpy as np
import matplotlib.pyplot as plt

# --- Estrapolazione metadati ---
from scipy.io import wavfile

fs, x = wavfile.read('/Users/generalkenobi/LocalDocuments/interazioni/Materiale/chirp_signal.wav')
# x = x.astype(np.float32) / np.iinfo(x.dtype).max

print(f'Frequenza di campionamento: {fs} Hz')
print(f'Numero di campioni: {len(x)}') # x NON è il numero di campioni, x SONO i campioni
print(f'Durata: {len(x)/fs:.2f} secondi')

durata = len(x)/fs

N = 8 # numero arbitrario
x_dec = x[::N] # ossia prendiamo un campione ogni 8
fnew = fs/N # troviamo

n_camp_orig = int(0.01 * fs)
n_camp_dec  = int(0.01 * fnew)  # campioni in 10ms nel segnale decimato

# Riproduciamo
import sounddevice as sd

print('Originale')
sd.play(x, samplerate=fs)
sd.wait()

print('Decimato')
sd.play(x_dec, samplerate=int(fnew))
sd.wait()

# Zoom temporale di 10ms

n_orig = int(0.01 * fs)      # campioni in 10ms - segnale originale
n_dec  = int(0.01 * fnew)  # campioni in 10ms - segnale decimato

# Segmento iniziale (bassa frequenza)
seg_orig_inizio = x[:n_camp_orig] # prendiamo tutti i campioni dall'inizio a 10 ms, dato che sappiamo quanti campioni ci sono in 10 ms basta prendere dal primo al "n_camp_orig"
seg_dec_inizio  = x_dec[:n_camp_dec] # uguale ma da 0 al numero di campioni in 10ms del campione decimato

# Segmento finale (alta frequenza)
seg_orig_fine = x[-n_camp_orig:] # qui invece prendiamo i campioni dei 10 ms finali dall'audio originale
seg_dec_fine  = x_dec[-n_camp_dec:] # campioni finali dell'audio decimato


# Assi temporali in millisecondi
t_orig = np.arange(n_orig) / fs * 1000
t_dec  = np.arange(n_dec)  / fnew * 1000


# Grafico

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle(f'Esercizio 3 – Downsampling N={N} (fs originale={fs} Hz → fs nuova={fnew:.0f} Hz)',
             fontsize=13)

# Riga 0: segmento iniziale
axes[0, 0].plot(t_orig, seg_orig_inizio, 'steelblue', lw=1.2)
axes[0, 0].set_title('Inizio – Originale (bassa freq)')
axes[0, 0].set_xlabel('Tempo (ms)')
axes[0, 0].set_ylabel('Ampiezza')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(t_dec, seg_dec_inizio, 'tomato', lw=1.2)
axes[0, 1].set_title(f'Inizio – Decimato N={N} (bassa freq)')
axes[0, 1].set_xlabel('Tempo (ms)')
axes[0, 1].set_ylabel('Ampiezza')
axes[0, 1].grid(alpha=0.3)

# Riga 1: segmento finale
axes[1, 0].plot(t_orig, seg_orig_fine, 'steelblue', lw=1.2)
axes[1, 0].set_title('Fine – Originale (alta freq)')
axes[1, 0].set_xlabel('Tempo (ms)')
axes[1, 0].set_ylabel('Ampiezza')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(t_dec, seg_dec_fine, 'tomato', lw=1.2)
axes[1, 1].set_title(f'Fine – Decimato N={N} (alta freq)')
axes[1, 1].set_xlabel('Tempo (ms)')
axes[1, 1].set_ylabel('Ampiezza')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

"""
Con frequenze basse i cicli sono lunghi, ci stanno comunque tanti
campioni anche dopo la decimazione. Con frequenze alte i cicli sono
cortissimi, e saltare 7 campioni su 8 significa perdere quasi tutta la
forma dell'onda. Notiamo nei grafici sovrastanti, che nonostante la
frequenza di campionamento cambi, la linea rimane uguale.
Negli ultimi 10ms, quando la frequenza è al suo massimo, vediamo che il
campionamento fa la differenza per non perdere dati
"""