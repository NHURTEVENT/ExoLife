import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('pics/Gliese_581d.pbm',0)
img2 = cv2.imread('pics/Gliese_581d V2.pbm',0)
img3 = copy.copy(img2)
img4 = copy.copy(img2)
img5 = copy.copy(img)
img6 = copy.copy(img2)
img7 = copy.copy(img2)


'''

cv2.normalize(img, img7, 255, 255, cv2.NORM_INF )
cv2.equalizeHist(img,img3)
originale = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="orginale")
#histo2 = plt.hist(img3.ravel(),256,[0,256],alpha=0.5,label="ega");
#histo1 = plt.hist(img7.ravel(),256,[0,256],alpha=0.5,label="norma");
cv2.imshow('3',img3)
cv2.imshow('7',img7)


isize = img2.shape

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if img4 [i][j] ==0 or img4 [i][j] >=252 :
            img4 [i][j] = 110

blur3 = cv2.medianBlur(img4,3)
cv2.imshow('4',img4)
cv2.imshow('blur3',blur3)
histo2 = plt.hist(img4.ravel(),256,[0,256],alpha=0.5,label="taitée");
plt.legend(loc='upper right')
plt.show()
'''
blur = cv2.medianBlur(img2,3)
blur2 = cv2.medianBlur(blur,3)
#blur = cv2.GaussianBlur(img2,(3,3),0)
#blur = cv2.bilateralFilter(img2,9,75,75)


'''
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


#fourier de l'image de base
plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')

#image restaurée par fourier  inverse
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Fourier inverse'), plt.xticks([]), plt.yticks([])

plt.show()



#On utilise une fonction pour enlever le bruit sur une image
img7 = cv2.fastNlMeansDenoising(img2,img7,3,7,21)


#on tente une transformation par cosinus discrète
imf= np.float32(img3)/255.0
dst = cv2.dct(imf)


#on coupe tout sauf le coin en haut d taille 200*200
dst[200:, :] = 0
dst[:, 200:] = 0


img6 = cv2.idct(dst)
#img6 = np.uint8(img6)*255.0

cv2.imshow('originale', img2)
cv2.imshow('dct', dst)
cv2.imshow('img4',img4)
cv2.imshow('img6',img6)
cv2.imshow('img7',img7)
'''
cv2.imshow('originale',img2)
cv2.imshow('blured', blur)
cv2.imshow('blured2', blur2)
cv2.waitKey(0)
cv2.destroyAllWindows()