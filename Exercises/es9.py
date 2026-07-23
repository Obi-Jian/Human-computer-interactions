import scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('./Materiale/gsr_signal.mat')

print(data.keys())

gsr = data['gsr'].flatten()
fs = 4

f_low = 1   # Hz - taglia il drift lento (sotto questa freq)
# filtro Butterworth di ordine 2
b, a = butter(2, f_low, btype='low', fs=fs)

gsr_cleaned = filtfilt(b, a, gsr)

f_low = 0.05
d, c = butter(2, f_low, btype='low', fs=fs)
scl = filtfilt(d, c, gsr_cleaned)

scr = gsr_cleaned - scl

t = np.arange(len(gsr)) / fs

# Punto 2: segnale grezzo
plt.figure(figsize=(12, 4))
plt.plot(t, gsr, color='steelblue')
plt.title('Segnale GSR grezzo')
plt.xlabel('Tempo (s)')
plt.ylabel('Conduttanza (μS)')
plt.grid(True)
plt.show()

# Punto 5: gsr_clean e SCL insieme
plt.figure(figsize=(12, 4))
plt.plot(t, gsr_cleaned, color='steelblue', label='GSR clean')
plt.plot(t, scl, color='red', linewidth=2, label='SCL (tonica)')
plt.title('Segnale GSR pulito e componente tonica (SCL)')
plt.xlabel('Tempo (s)')
plt.ylabel('Conduttanza (μS)')
plt.legend()
plt.grid(True)
plt.show()

# Punto 6: SCR separata
plt.figure(figsize=(12, 4))
plt.plot(t, scr, color='green')
plt.title('Componente fasica (SCR)')
plt.xlabel('Tempo (s)')
plt.ylabel('Conduttanza (μS)')
plt.grid(True)
plt.show()

distance_min = int(1 * fs)  # = 4 campioni
peaks, _ = find_peaks(scr, height=0.01, distance=distance_min)

rr_campioni = np.mean(np.diff(peaks))
rr_secondi = rr_campioni / fs

# Visualizzazione picchi
plt.figure(figsize=(12, 4))
plt.plot(t, scr, color='green', label='SCR')
plt.plot(t[peaks], scr[peaks], '^', color='red', markersize=8, label='Picchi')
plt.title(f'Picchi SCR rilevati: {len(peaks)}')
print(f"Intervallo R-R medio: {rr_secondi:.3f} s")
plt.xlabel('Tempo (s)')
plt.ylabel('Conduttanza (μS)')
plt.legend()
plt.grid(True)
plt.show()

# Metriche toniche (SCL)
scl_mean = np.mean(scl)
scl_std = np.std(scl)
print(f"SCL media: {scl_mean:.4f} μS")
print(f"SCL deviazione standard: {scl_std:.4f} μS")

# Metriche fasiche (SCR)
ampiezze = scr[peaks]
print(f"Numero picchi: {len(peaks)}")
print(f"Ampiezza media: {np.mean(ampiezze):.4f} μS")
print(f"Ampiezza massima: {np.max(ampiezze):.4f} μS")
print(f"Somma totale ampiezze: {np.sum(ampiezze):.4f} μS")