import numpy as np
import matplotlib.pyplot as plt

# --- Parametri ---
f_reale = 100
fs = 150
f_alias = abs(fs - f_reale)  # 50 Hz

# --- Vettori temporali ---
fs_alta = 10000
t_reale = np.arange(0, 0.1, 1/fs_alta) # il range è da 0 a 0.1 secondi, la frequenza con cui misura è 10k volte al secondo
t_camp  = np.arange(0, 0.1, 1/fs) # arrange dice: salva tutti i numeri che ci sono tra 0 e 0.1 ogni 1/fs numeri
# restituisce un array da 0.1/(1/1000) = 0.1/0.001 = 1000 elementi

# --- Segnali ---
y_reale = np.sin(2 * np.pi * f_reale * t_reale) # y sarebbe un array di sinusoidi valutate in un tot di t
y_camp  = np.sin(2 * np.pi * f_reale * t_camp)
y_alias = -np.sin(2 * np.pi * f_alias * t_reale)

# --- Grafico ---
plt.figure(figsize=(10, 5))
plt.plot(t_reale, y_reale, 'k',    linewidth=1,   label='Segnale reale (100 Hz)')
plt.plot(t_reale, y_alias, 'b--',  linewidth=2,   label=f'Alias percepito ({f_alias} Hz)')
plt.stem(t_camp,  y_camp,
         linefmt='r-', markerfmt='ro', basefmt='k-')
plt.gca().get_lines()[-1].set_label('Campioni presi')  # etichetta per stem

plt.title(f'Effetto Aliasing: campionando a {fs} Hz')
plt.xlabel('Tempo (s)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()