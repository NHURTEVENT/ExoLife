import numpy as np
import cv2

#Récupération de l'image
img = cv2.imread('pics/Encelade_surface.pbm',0)

isize = img.shape

print(img.shape)

max =0 #de plus ne plus blanc
compteur =0 #index du tableau de coord
coords = []

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if img[i][j] > max:
            coords.clear()
            max = img[i][j]
            coord = [i, j]
            coords.append(coord)
        elif img[i][j] == max:
            max = img[i][j]
            coord = [i, j, max]
            coords.append(coord)


for coord in coords :
    img = cv2.circle(img, (coord[0], coord[1]), 10, (255,0,0), 1) ##On entoure donc la valeur la plus blanche de l'image
    print(coord)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()