import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('pics/Gliese_581d.pbm',0)
img2 = cv2.imread('pics/Gliese_581d V2.pbm',0)
img3 = copy.copy(img2)
img5 = copy.copy(img)

blur = cv2.medianBlur(img2,3)
blur2 = cv2.medianBlur(blur,3)
#blur = cv2.GaussianBlur(img2,(3,3),0)
#blur = cv2.bilateralFilter(img2,9,75,75)

#on efectue un transformé de fourier sur l'image
f = np.fft.fft2(img2)
#met la fréquence 0 au centre
#fshift = np.fft.fftshift(f)
fshift=f

#on récupère la taille de l'image
rows, cols = img2.shape
#on récupère les coord du milieu
crow,ccol = int(rows/2) , int(cols/2)

#on place un carré de 40 par 40 au centre = on filtre les hautes fréquences
fshift[crow-10:crow+10, ccol-10:ccol+10] = 1


#On fait le transformé inverse
img_back = np.fft.ifft2(fshift)
img_back = np.abs(img_back)

#On le plot logarithmiquement
magnitude_spectrum = np.log(np.abs(fshift))

'''
#fourier de l'image de base
plt.subplot(111),plt.imshow(magnitude_spectrum, cmap = 'gray')
'''

'''
dst = cv2.fastNlMeansDenoising(img2,img3,3,7,21)
plt.subplot(121),plt.imshow(img2, cmap = 'gray')
plt.title('originale'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst, cmap = 'gray')
plt.title('denoised'), plt.xticks([]), plt.yticks([])
'''

'''
#image restaurée par fourier  inverse
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Fourier inverse'), plt.xticks([]), plt.yticks([])
'''
plt.show()

cv2.dct(img2,img3,cv2.DCT_ROWS)

cv2.imshow('originale', img2)
cv2.imshow('dct', img3)
cv2.imshow('blured', blur)
cv2.imshow('blured2', blur2)
cv2.waitKey(0)
cv2.destroyAllWindows()