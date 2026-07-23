""" Dato un segnale sinusoidale x(t)=Asin(2πft) con ampiezza A=1 e frequenza f=50 Hz
1. Calcolare il numero di livelli di quantizzazione disponibili per una risoluzione di 2 bit e 4 bit.
2. Determinare lo step di quantizzazione Δ
Implementazione in Python/MATLAB:
1. Generare il segnale originale con una frequenza di campionamento di 1000 Hz
per una durata di 0.1 s.
2. Implementare una funzione di quantizzazione uniforme che mappi il segnale
continuo sui livelli discreti.
3. Visualizzare il segnale quantizzato, mettendolo a confronto con l'originale. """

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# --- Estrapolazione metadati ---
f = 50
samplerate = 1000
durata = 0.1
A = 1
t = np.arange(0, durata, 1/samplerate)
y = A * np.sin(2 * np.pi * f * t) # potevamo non moltiplicare per A in quanto il seno oscilla tra 1 e -1, dunque la sua ampiezza è 1 di base

liv_quant_2bit = 4 # 2^n_bit
liv_quant_4bit = 16

# step di quantizzazione
delta_2bit = A*2 / (liv_quant_2bit -1) # (liv max - liv min) / n livelli = (1-(-1))/4 = 0.5
delta_4bit = A*2 / (liv_quant_4bit -1) # (1-(-1))/16 = 1/8 = 0.125

# quantizziamo
y_2bit = np.round(y / delta_2bit) * delta_2bit
y_4bit = np.round(y / delta_4bit) * delta_4bit

#oppure possiamo creare una formula di quantizzazione che arrotonda automaticamente ogni valore di y al valore più vicino a nostra disposizione (asse y) in base alla risoluzione
def quantizza(segnale, n_bit, A_min=-1, A_max=1):
    livelli = 2 ** n_bit
    delta = (A_max - A_min) / (livelli - 1)  # we also can do without the -1
    y_quant = np.round(segnale / delta) * delta
    return y_quant

# non è necessario, ma ricalcoliamo y quantizzato con la funzione apposita
y_2bit = quantizza(y, 2)
y_4bit = quantizza(y, 4)

# analisi quantizzazione
"""
1. Calcolare il Segnale Errore (differenza istantanea tra segnale originale e
quantizzato).
2. Calcolare l'Errore Medio Quadratico (MSE) per entrambe le risoluzioni e
commentare come cambia la precisione al raddoppiare dei bit.
3. Se aumentiamo la risoluzione di 1 solo bit (es. da 4 a 5 bit), di quanti decibel
(dB) ci aspettiamo che migliori il rapporto segnale-rumore?
"""

errore_2bit = y - y_2bit # numPy automatically subtracts the value of each element in the array
errore_4bit = y - y_4bit

# note: the signal error for each element can't be higher than delta/2 or lower than -(delta/2)

# MSE
mse_2bit = np.mean(errore_2bit ** 2)
mse_4bit = np.mean(errore_4bit ** 2)

print(f'Scarto quadratico medio 2 bit: {mse_2bit}')
print(f'Scarto quadratico medio 4 bit: {mse_4bit}')
# nota: a 4 bit si riduce notevolmente

#proviamo con 5 bit
y_5bit = quantizza(y, 5)
errore_5bit = y - y_5bit
mse_5bit = np.mean(errore_5bit ** 2)
print(f'Scarto quadratico medio 5 bit: {mse_5bit}')


# calcolo rapporto segnale - rumore in dB
# SNR (dB) = 10 * log10(potenza segnale / potenza rumore)
snr_2bit = 10 * np.log10(np.mean(y**2) / np.mean(errore_2bit**2))
snr_4bit = 10 * np.log10(np.mean(y**2) / np.mean(errore_4bit**2))
snr_5bit = 10 * np.log10(np.mean(y**2) / np.mean(errore_5bit**2))


print(f'SNR 2 bit: {snr_2bit:.2f} dB')
print(f'SNR 4 bit: {snr_4bit:.2f} dB')
print(f'SNR 5 bit: {snr_5bit:.2f} dB')


# Grafici
for n_bit, y_quant, errore, colore in [
    (2, y_2bit, errore_2bit, 'tomato'),
    (4, y_4bit, errore_4bit, 'seagreen'),
    (5, y_5bit, errore_5bit, 'steelblue')
]:
    delta = 2 / ((2 ** n_bit) -1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f'Quantizzazione {n_bit} bit', fontsize=13)

    # Segnale quantizzato
    axes[0].plot(t, y, 'lightgray', lw=1, label='originale')
    axes[0].plot(t, y_quant, color=colore, lw=1.2, label=f'{n_bit} bit ({2**n_bit} livelli)')
    axes[0].set_title('Segnale quantizzato')
    axes[0].set_ylabel('Ampiezza')
    
    # per 5 bit: linee ogni delta ma label solo ogni 0.125 (come 4 bit)
    livelli = np.arange(-1, 1 + delta, delta)
    axes[0].set_yticks(livelli)
    if n_bit == 5:
        label_ticks = np.arange(-1, 1 + 0.125, 0.125)
        axes[0].set_yticklabels([f'{v:.3f}' if any(np.isclose(v, label_ticks)) else '' for v in livelli])

    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Errore
    axes[1].plot(t, errore, color=colore, lw=1.2)
    axes[1].axhline(y=0, color='black', lw=0.8)
    axes[1].axhline(y=delta/2,  color='gray', lw=0.8, linestyle='--', label=f'±Δ/2 = ±{delta/2:.4f}')
    axes[1].axhline(y=-delta/2, color='gray', lw=0.8, linestyle='--')
    axes[1].set_title('Errore di quantizzazione')
    axes[1].set_xlabel('Tempo (s)')
    axes[1].set_ylabel('Errore')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

plt.show()