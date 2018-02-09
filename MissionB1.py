import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pics/Gliese 667Cc_surface.pbm',0)
img2 = copy.copy(img)
img3 = copy.copy(img)
img4 = copy.copy(img)
img5 = copy.copy(img)



plt.show()

#g(x,y) = (f(x,y) - fmin)*2^bpp / (fmax-fmin)
#normalisation linéaire img2 = (img1-min)* (newMax-newMin)/(max-min) + newMin

#égalisation
#calc histo, normalise à 255, calcule l'intégrale de l'histogramme H', utilise l'intégralle pour remaper l'histogramme dst(x,y) = H'(src(x,y))


cv2.normalize(img, img2, 255, 255, cv2.NORM_INF )
cv2.equalizeHist(img,img3)
#cv2.normalize(img3, img4, 255, 255, 1 )
#cv2.equalizeHist(img2,img5)
#cv2.imshow("originale", img)
#cv2.imshow("norma", img2)
cv2.imshow("equa", img3)
#cv2.imshow("equa-norma", img4)
#cv2.imshow("norma-equa", img5)

originale = plt.hist(img.ravel(),256,[0,256],alpha=0.5,label="orginale");
histo1 = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="norma");
histo2 = plt.hist(img3.ravel(),256,[0,256],alpha=0.5,label="ega");
#histo3 = plt.hist(img4.ravel(),256,[0,256],alpha=0.5,label="equ-norma");
#histo4 = plt.hist(img5.ravel(),256,[0,256],alpha=0.5,label="norma-equa");


plt.legend(loc='upper right')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()