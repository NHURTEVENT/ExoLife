import numpy as np
import cv2
import math

#Récupération de l'image
img = cv2.imread('pics/Europa_surface.pbm',0)
isize = img.shape
"""
window = int(math.sqrt(img.size))
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

img2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,10001,0)

max = 200

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if img[i][j] < max :
            img[i][j] =0
        else    :
            img[i][j] =255



cv2.imshow("hard", img)
cv2.imshow("image2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
