#FFT de deux sons purs (440 Hz + 880 Hz)
import numpy as np
import matplotlib.pyplot as plt
fs = 44100
T = 1
t = np.arange(0, T, 1/fs)

f1 = 440
f2 = 880

x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

# FFT
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(X), 1/fs)

# Module FFT
plt.figure()
plt.plot(freqs, np.abs(X))
plt.xlim(0, 2000)
plt.title("Spectre du signal (FFT)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

#Ajout et filtrage du bruit (5000 Hz)
f_bruit = 5000
bruit = 0.3 * np.sin(2*np.pi*f_bruit*t)
x_bruite = x + bruit

Xb = np.fft.fft(x_bruite)

# Filtrage passe-bas simple
Xf = Xb.copy()
Xf[np.abs(freqs) > 2000] = 0

# Signal filtré
x_filtre = np.fft.ifft(Xf)

# Tracé
plt.figure()
plt.plot(freqs, np.abs(Xb))
plt.xlim(0, 6000)
plt.title("Spectre du signal bruité")
plt.show()

plt.figure()
plt.plot(t, x, label="Original")
plt.plot(t, np.real(x_filtre), label="Filtré", alpha=0.7)
plt.legend()
plt.title("Comparaison signal original / filtré")
plt.show()