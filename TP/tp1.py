#Génération d’un signal sinusoïdal
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
A = 1
f0 = 10
phi = 0
fs = 100
T = 1

# Vecteur temps
t = np.arange(0, T, 1/fs)

# Signal sinusoïdal
x = A * np.sin(2*np.pi*f0*t + phi)

# Tracé
plt.figure()
plt.plot(t, x)
plt.title("Signal sinusoïdal x(t)")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Bruit blanc gaussien
bruit = np.random.normal(0, 0.3, len(x))

# Signal bruité
y = x + bruit

# Tracé
plt.figure()
plt.plot(t, x, label="Signal pur")
plt.plot(t, y, label="Signal bruité", alpha=0.7)
plt.legend()
plt.title("Signal pur et signal bruité")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Signal porte
N = 100
rect = np.zeros(N)
rect[20:40] = 1

# Convolution
conv_rect = np.convolve(rect, rect)

# Tracés
plt.figure()
plt.stem(rect)
plt.title("Signal porte")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.show()

plt.figure()
plt.stem(conv_rect)
plt.title("Convolution du signal porte avec lui-même")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.show()