import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

#image en B&W
img = cv2.imread('pics/U1_surface.pbm', 0)
img2 = copy.copy(img)
img3 = copy.copy(img)
img4 = copy.copy(img)
img5 = copy.copy(img)
img6 = copy.copy(img)

cv2.normalize(img, img2, 255, 255, cv2.NORM_INF )
cv2.equalizeHist(img,img3)
#On calcule l'histogramme de l'originale
originale = plt.hist(img.ravel(),256,[0,256],alpha=0.5,label="orginale")
histo1 = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="norma");
histo2 = plt.hist(img3.ravel(),256,[0,256],alpha=0.5,label="ega");

plt.show()

laplacian = cv2.Laplacian(img5,cv2.CV_64F)
sobelx = cv2.Sobel(img5,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img5,cv2.CV_64F,0,1,ksize=5)
sobelxy = cv2.Sobel(img5,cv2.CV_64F,1,1,ksize=5)
sobelxyz = copy.copy(sobelxy)
cv2.normalize(laplacian, laplacian, 255, 255, cv2.NORM_INF )
cv2.normalize(sobelxy, sobelxyz, 255, 255, cv2.NORM_INF )

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobelxyz,cmap = 'gray')
plt.title('Sobel XYZ'), plt.xticks([]), plt.yticks([])

plt.show()

histo3 = plt.hist(laplacian.ravel(),256,[0,256],alpha=0.5,label="lap");
histo4 = plt.hist(sobelx.ravel(),256,[0,256],alpha=0.5,label="sx");
histo5 = plt.hist(sobelxy.ravel(),256,[0,256],alpha=0.5,label="sxy");
histo6 = plt.hist(sobely.ravel(),256,[0,256],alpha=0.5,label="sy");
histo7 = plt.hist(sobelxyz.ravel(),256,[0,256],alpha=0.5,label="sxyz");
plt.legend(loc = 'upper right')
plt.show()






cv2.imshow('img',img)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)







#nombre de pixels dans l'iamge
isize = img.shape

#on efectue un transformé de fourier sur l'image
f = np.fft.fft2(img)
#met la fréquence 0 au centre
#fshift = np.fft.fftshift(f)
fshift=f
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
fshift2[crow-200:crow+200, ccol-200:ccol+200] = 1


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

isize = img.shape

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if sobelxyz[i][j] < 20 :
            sobelxyz[i][j] = 0


cv2.imshow('img7',sobelxyz)

cv2.waitKey(0)
cv2.destroyAllWindows()