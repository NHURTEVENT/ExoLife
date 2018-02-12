import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt

#image en B&W
img = cv2.imread('pics/HD215497.pbm', 0)
#image chargée en couleur pour pouvoir la colorier ensuite
img2 = cv2.imread('pics/HD215497.pbm',1)

#Copie de l'image en noir et blanc
img3= copy.copy(img)
#Copies de l'image couleur
img4= copy.copy(img2)
img5= copy.copy(img2)

#nombre de pixels dans l'iamge
isize = img.shape

#On calcule l'histogramme de l'originale
originale = plt.hist(img.ravel(),256,[0,256],alpha=0.5,label="orginale")

#on normalise = on contraste
#normalisation linéaire img2 = (img1-min)* (newMax-newMin)/(max-min) + newMin
cv2.normalize(img, img3, 255, 255, cv2.NORM_INF )

#On égalise
cv2.equalizeHist(img,img3)
cv2.equalizeHist(img,img4)

#On calcule les histogrammes des images normalisées et égalisées
histo1 = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="norma");
histo2 = plt.hist(img3.ravel(),256,[0,256],alpha=0.5,label="ega");


#Essai 1 : quadruple threasholding avec intervale réguliers
thresh1 = 64
thresh2=128
thresh3=192

#on parcourt chaque pixel
for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        #Si il est inférieur au plus petit threshold
        if img[i][j] <= thresh3 :
            if img[i][j] <= thresh2 :
                if img[i][j] <= thresh1:
                    #img[i][j] = 0
                    #Si il est plus grand que 192 on le coloie en noir
                    img2[i][j] = [0,0,0]
                #Si il est plus petit (entre 64 et 128) on le colorie en bleu
                else :
                    #img[i][j] = 64
                    img2[i][j] = [0, 0, 255]
            # Si il est plus petit (entre 128 et 192) on le colorie en rouge
            else :
                #img[i][j] = 128
                img2[i][j] = [255, 0, 0]
        #Si il est plus petit on le colorie en jaune
        else    :
            #img[i][j] =192
            img2[i][j] = [0, 255, 255]


seuil = []
sepa=4
#Variable pour stocker l'aire de 0 à x
Afx =0
#Aire de 0 au treshold précédent
Aprec = 0
#Aire totale de l'intégrale de l'histogramme
threshold = sum(originale[0][0:254])/sepa




#de 0 à 254
for x in range(0,originale[0].size-1) :

#on calcule l'intégrale 0,x de l'histogram

    Afx = sum(originale[0][0:int(x)])
    print('Aprec')
    print(Aprec)
#Si cette intégralle vaut 1/4 ou plus de l'intégralle totale
    if (Afx-Aprec)>= threshold :
        #On set notre intégrale précedente avec l'actuelle
        Aprec = Afx
        print('Afx')
        print(Afx)
        #On ajoute ce x comme seuil
        seuil.append(x)


#on affiche les seuils
for i in range(0,sepa-1) :
    print('seuil')
    print(seuil[i])

#On parcourt chaque pixel de l'iamge
for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        #Si il est infériur au plus grand seuil
        if img[i][j] <= seuil[2] :
            if img[i][j] <= seuil[1] :
                if img[i][j] <= seuil[0]:
                    #si il est plus petit que le plus petit seuil on le colorie en noir
                    img5[i][j] = [0,0,0]
                #si il est entre le plus petit seuil et celui d'après on le colorie en rouge
                else :
                    img5[i][j] = [0, 0, 255]
            #Si il est entre le plus grand seuil et celui d'avant on le colorie en bleu
            else :
                img5[i][j] = [255, 0, 0]
        #Si il est plus grand que le plus grand seuil on le colorie en jaune
        else    :
            img5[i][j] = [0, 255, 255]


print(Afx)
print(img2.size)
cv2.imshow("originale", img)
#Cette version est plus esthétique mais moins juste
#cv2.imshow("normaColo", img2)
#cv2.imshow("ega", img3)
#cv2.imshow("norma", img4)
cv2.imshow("integrale", img5)
plt.legend(loc='upper right')
plt.show()