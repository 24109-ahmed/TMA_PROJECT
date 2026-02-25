#Partie 1 : Aliasing audio
from scipy.io import wavfile
import cv2

fs, audio = wavfile.read("audio.wav")

# Sous-échantillonnage sauvage
audio_alias = audio[::10]
fs_alias = fs // 10

wavfile.write("audio_alias.wav", fs_alias, audio_alias)

#Partie 2 : Image & Quantification

img = cv2.imread("image.png", 0)

img_2bits = (img // 64) * 64
img_1bit = (img > 128) * 255

cv2.imshow("Original", img)
cv2.imshow("2 bits", img_2bits.astype(np.uint8))
cv2.imshow("1 bit", img_1bit.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()