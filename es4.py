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
delta_2bit = A*2 / liv_quant_2bit # (liv max - liv min) / n livelli = (1-(-1))/4 = 0.5
delta_4bit = A*2 / liv_quant_4bit # (1-(-1))/16 = 1/8 = 0.125

y_2bit = np.round(y / delta_2bit) * delta_2bit
y_4bit = np.round(y / delta_4bit) * delta_4bit

#oppure possiamo creare una formula di quantizzazione che arrotonda ogni valore di y al valore più vicino a nostra disposizione (asse y) in base alla risoluzione
def quantizza(segnale, n_bit):
    livelli = 2 ** n_bit
    delta = 2 / livelli  # range totale = 2 (da -1 a +1)
    
    # arrotonda ogni campione al livello più vicino
    y_quant = np.round(segnale / delta) * delta
    
    return y_quant

y_2bit = quantizza(y, 2)
y_4bit = quantizza(y, 4)

# grafici
"""
fig, axes = plt.subplots(3, 1, figsize=(10, 12), 
    gridspec_kw={'height_ratios': [1, 1, 1.5]}) 
"""
# eventualmente possiamo migliorare la visualizzazione nell'ultimo grafico sostituendo la prossima linea con queste due
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Quantizzazione uniforme', fontsize=13)

axes[0].plot(t, y, 'steelblue', lw=1.2)
axes[0].set_title('Segnale originale')
axes[0].set_ylabel('Ampiezza')
axes[0].grid(alpha=0.3)

livelli_2bit = np.arange(-1, 1 + delta_2bit, delta_2bit)
axes[2].set_yticks(livelli_2bit)
axes[1].plot(t, y, 'lightgray', lw=1, label='originale')
axes[1].plot(t, y_2bit, 'tomato', lw=1.2, label='2 bit (4 livelli)')
axes[1].set_title('Quantizzato 2 bit')
axes[1].set_ylabel('Ampiezza')
axes[1].legend()
axes[1].grid(alpha=0.3)

livelli_4bit = np.arange(-1, 1 + delta_4bit, delta_4bit)
axes[2].set_yticks(livelli_4bit)
axes[2].plot(t, y, 'lightgray', lw=1, label='originale')
axes[2].plot(t, y_4bit, 'seagreen', lw=1.2, label='4 bit (16 livelli)')
axes[2].set_title('Quantizzato 4 bit')
axes[2].set_xlabel('Tempo (s)')
axes[2].set_ylabel('Ampiezza')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# analisi quantizzazione