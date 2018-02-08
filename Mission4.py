import numpy as np
import cv2
import math
import copy
from matplotlib import pyplot as plt

#Récupération de l'image
img = cv2.imread('pics/Jupiter1.pbm',0)
img2 = cv2.imread('pics/Jupiter2.pbm',0)
isize = img.shape
isize2 = img2.shape
img3 = [[]]

"""
window = in1t(math.sqrt(img.size))
height = img.shape[0]
width = img.shape[1]
print(window)
print(" ")
print(height)
print(" ")
print(width)
print(" ")
print(img.size)
"""
print(img.shape[0])
print(" ")
print(img.shape[1])
print(" ")
print(img2.shape[0])
print(" ")
print(img2.shape[1])
img3 = copy.copy(img)



for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        var1 = img[i][j]
        var2 = img2[i][j]
        img3[i][j] = int((int(var1) + int(var2) )/2) # on prends la moyenne des deux images


kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
cv2.imshow("img", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()