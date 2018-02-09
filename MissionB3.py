import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pics/HD215497.pbm', 0)
img2 = cv2.imread('pics/HD215497.pbm',1)

img3= copy.copy(img)
img4= copy.copy(img2)
isize = img.shape

originale = plt.hist(img.ravel(),256,[0,256],alpha=0.5,label="orginale")

#on normalise = on contraste
#normalisation linéaire img2 = (img1-min)* (newMax-newMin)/(max-min) + newMin
cv2.normalize(img, img3, 255, 255, cv2.NORM_INF )
cv2.equalizeHist(img,img3)
histo1 = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="norma");
histo2 = plt.hist(img3.ravel(),256,[0,256],alpha=0.5,label="ega");

thresh1 = 64
thresh2=128
thresh3=192

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if img[i][j] < thresh3 :
            if img[i][j] < thresh2 :
                if img[i][j] < thresh1:
                    img[i][j] = 0
                    img2[i][j] = [0,0,0]
                else :
                    img[i][j] = 64
                    img2[i][j] = [0, 0, 255]
            else :
                img[i][j] = 128
                img2[i][j] = [255, 0, 0]
        else    :
            img[i][j] =192
            img2[i][j] = [0, 255, 255]


cv2.imshow("originale", img)
cv2.imshow("norma", img2)
cv2.imshow("ega", img3)
cv2.imshow("4", img4)
plt.legend(loc='upper right')
plt.show()