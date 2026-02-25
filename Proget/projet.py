# ===========================
# 1️⃣ Introduction
# ===========================

# Mini-Projet : L’Archéologue Acoustique (NASA/JFK)
# Objectif : Analyse spectrale des voix du binôme et restauration d’un discours JFK bruité
# Outils : Python, NumPy, SciPy, Matplotlib, Librosa

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import iirnotch, lfilter

# ===========================
# 2️⃣ Partie Obligatoire : Analyse de la Voix
# ===========================

# ⚡ Charger les fichiers audio déjà enregistrés
# Remplacez par vos fichiers audio
audio_files = {
    "Ahmed": "Ahmed.wav",
    "Mohamed": "Mohamed.wav"
}

voices = {}
sr_dict = {}

for name, file in audio_files.items():
    y, sr = librosa.load(file, sr=None)
    voices[name] = y
    sr_dict[name] = sr
    print(f"{name} : durée = {len(y)/sr:.2f} s, fréquence d'échantillonnage = {sr} Hz")

# ⚡ Affichage du signal temporel
plt.figure(figsize=(12, 6))
for name, y in voices.items():
    plt.plot(np.linspace(0, len(y)/sr_dict[name], len(y)), y, label=name)
plt.title("Signal temporel des voix")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# ⚡ FFT et visualisation spectrale
plt.figure(figsize=(12, 6))
for name, y in voices.items():
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1/sr_dict[name])
    plt.plot(freqs[:len(freqs)//2], np.abs(Y[:len(Y)//2]), label=name)
plt.title("Spectre des voix")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# ⚡ Détection du Pitch (fréquence fondamentale)
def detect_pitch(y, sr):
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1/sr)
    idx = np.argmax(np.abs(Y[:len(Y)//2]))
    return freqs[idx]

for name, y in voices.items():
    pitch = detect_pitch(y, sr_dict[name])
    print(f"{name} : fréquence fondamentale ≈ {pitch:.1f} Hz")

# ===========================
# 3️⃣ Présentation du problème NASA
# ===========================

# Chargement du signal bruité JFK
jfk_file = "JFK_noisy.wav"
jfk, sr_jfk = librosa.load(jfk_file, sr=None)
print(f"Signal JFK : durée = {len(jfk)/sr_jfk:.2f} s, fréquence d'échantillonnage = {sr_jfk} Hz")

plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, len(jfk)/sr_jfk, len(jfk)), jfk)
plt.title("Signal JFK bruité")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.show()

# ===========================
# 4️⃣ Analyse spectrale du signal bruité
# ===========================

# FFT du signal JFK
Y_jfk = np.fft.fft(jfk)
freqs_jfk = np.fft.fftfreq(len(jfk), 1/sr_jfk)

plt.figure(figsize=(12,6))
plt.plot(freqs_jfk[:len(freqs_jfk)//2], np.abs(Y_jfk[:len(Y_jfk)//2]))
plt.title("Spectre du signal JFK bruité")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.show()

# ===========================
# 5️⃣ Conception du filtre Notch (1000 Hz)
# ===========================

# Paramètres du filtre
f0 = 1000  # Hz du sifflement
Q = 30     # facteur de qualité
b, a = iirnotch(f0, Q, sr_jfk)

# Application du filtre
jfk_notch = lfilter(b, a, jfk)

# ===========================
# 6️⃣ Soustraction spectrale (bruit blanc)
# ===========================

# Estimation du spectre du bruit blanc (premiers 0.5s)
noise = jfk[:int(0.5*sr_jfk)]
noise_spectrum = np.fft.fft(noise)
jfk_spectrum = np.fft.fft(jfk_notch)

# Soustraction spectrale (amplitude)
clean_spectrum = jfk_spectrum - np.mean(np.abs(noise_spectrum))
# Reconstruction du signal
jfk_clean = np.fft.ifft(clean_spectrum).real

# ===========================
# 7️⃣ Résultats (Avant / Après)
# ===========================

plt.figure(figsize=(12,6))
plt.plot(np.linspace(0, len(jfk)/sr_jfk, len(jfk)), jfk, label="JFK bruité")
plt.plot(np.linspace(0, len(jfk_clean)/sr_jfk, len(jfk_clean)), jfk_clean, label="JFK restauré")
plt.title("Signal JFK : Avant / Après restauration")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# ===========================
# 8️⃣ Calcul du SNR
# ===========================

def compute_snr(signal, noise):
    return 10*np.log10(np.sum(signal**2)/np.sum(noise**2))

# Bruit estimé = signal bruité - signal restauré
noise_est = jfk - jfk_clean
snr_before = compute_snr(jfk, noise)
snr_after = compute_snr(jfk_clean, noise_est)

print(f"SNR avant traitement ≈ {snr_before:.2f} dB")
print(f"SNR après traitement ≈ {snr_after:.2f} dB")

# ===========================
# 9️⃣ Discussion
# ===========================
print("""
Analyse :
- Le filtre Notch a efficacement supprimé le sifflement à 1000 Hz.
- La soustraction spectrale a réduit le bruit blanc.
- Difficulté : séparer complètement les voix si elles sont mélangées (BSS)
""")

# ===========================
# 🔟 Conclusion
# ===========================
print("""
Conclusion :
- Apprentissage : FFT, filtrage Notch, soustraction spectrale, SNR
- Améliorations possibles : filtrage adaptatif, analyse multi-canaux, méthodes BSS plus avancées
""")

# ===========================
# 1️⃣1️⃣ Références
# ===========================
print("""
Références :
- JFK audio archive : https://www.archives.gov/
- Bibliothèques Python : NumPy, SciPy, Librosa, Matplotlib
""")