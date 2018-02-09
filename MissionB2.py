import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pics/GD61.pbm', 0)
img2= copy.copy(img)
img3= copy.copy(img)

originale = plt.hist(img.ravel(),256,[0,256],alpha=0.5,label="orginale")

#on normalise = on contraste
#normalisation linéaire img2 = (img1-min)* (newMax-newMin)/(max-min) + newMin
cv2.normalize(img, img2, 255, 255, cv2.NORM_INF )
histo1 = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="norma");


cv2.imshow("originale", img)
cv2.imshow("norma", img2)
plt.legend(loc='upper right')
plt.show()