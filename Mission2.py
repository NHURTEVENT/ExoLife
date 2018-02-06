import numpy as np
import cv2

#Récupération de l'image
img = cv2.imread('pics/Mars_surface.pbm',0)

isize = img.shape
somme =0 #somme de toutes les quantitées de gaz présente dans l'image

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        somme+= img[i][j]
nbPixel = img.size #nombre de pixels dans l'image
gasDensity = somme/(nbPixel*255) #la densitée de gaz est la quantité totale de gaz divisée par l'aire (le nb de pixels) divisé par 255 car 255=100%
print("densité: %f %%" %gasDensity)
