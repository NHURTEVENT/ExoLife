import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

#image en B&W
img = cv2.imread('pics/tardisBleu.png', 0)
img2 = cv2.imread('pics/tardis.png',0)


#nombre de pixels dans l'iamge
isize = img.shape

#on efectue un transformé de fourier sur l'image
f = np.fft.fft2(img2)
#met la fréquence 0 au centre
fshift = np.fft.fftshift(f)

#On le plot logarithmiquement
magnitude_spectrum = np.log(np.abs(fshift))

#On fait le transformé inverse
img_back = np.fft.ifft2(fshift)
img_back = np.abs(img_back)

#on récupère la taille de l'image
rows, cols = img2.shape
#on récupère les coord du milieu
crow,ccol = int(rows/2) , int(cols/2)

#On copie notre fourier
fshift2 = copy.copy(fshift)

#on place un carré de 40 par 40 au centre = on filtre les hautes fréquences
fshift2[crow-20:crow+20, ccol-20:ccol+20] = 1


#On plot le transformé inverse logarithmiquement
magnitude_spectrum2 = np.log(np.abs(fshift2))

#On fait le transformé inverse
img_back2 = np.fft.ifft2(fshift2)
img_back2 = np.abs(img_back2)

#image de base
plt.subplot(151),plt.imshow(img2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

#fourier de l'image de base
plt.subplot(152),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])

#image restaurée par fourier  inverse
plt.subplot(153),plt.imshow(img_back, cmap = 'gray')
plt.title('Fourier inverse'), plt.xticks([]), plt.yticks([])

#fourier avec le filtre
plt.subplot(154),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Fourier filtré'), plt.xticks([]), plt.yticks([])

#image restaurée par fourier inverse avec celui filtré
plt.subplot(155),plt.imshow(img_back2, cmap = 'gray')
plt.title('Fourier filtré inverse'), plt.xticks([]), plt.yticks([])

plt.show()