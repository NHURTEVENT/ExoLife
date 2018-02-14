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
img7 = copy.copy(img)

#nombre de pixels dans l'iamge
isize = img.shape

cv2.normalize(img, img2, 255, 255, cv2.NORM_INF )
cv2.equalizeHist(img,img3)
#On calcule l'histogramme de l'originale
originale = plt.hist(img.ravel(),256,[0,256],alpha=0.5,label="orginale")
histo1 = plt.hist(img2.ravel(),256,[0,256],alpha=0.5,label="norma");
histo2 = plt.hist(img3.ravel(),256,[0,256],alpha=0.5,label="ega");

plt.show()

'''
#filtrage de canny
edges = cv2.Canny(img,100,210)
cv2.imshow('edge',edges)
#laplacien
laplacian = cv2.Laplacian(img5,cv2.CV_64F)

#sobel
sobelx = cv2.Sobel(img5,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img5,cv2.CV_64F,0,1,ksize=5)
sobelxy = cv2.Sobel(img5,cv2.CV_64F,1,1,ksize=5)
sobelxyz = copy.copy(sobelxy)

#on tente de normaliser les résultats
cv2.normalize(laplacian, laplacian, 255, 255, cv2.NORM_INF )

cv2.normalize(sobelxy, sobelxyz, 255, 255, cv2.NORM_INF )
sobelxy2 = copy.copy(sobelxyz)

#on affiche les images obtenues
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobelxyz,cmap = 'gray')
plt.title('Sobel XYZ'), plt.xticks([]), plt.yticks([])

plt.show()

#on affiche les histogrammes 
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






#on test un traitement à l'aide de la transformée de fourier


#on effectue une transformée de fourier sur l'image
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
#on récupère les coord du milieu pour appliquer des filtres
crow,ccol = int(rows/2) , int(cols/2)

#On copie notre fourier
fshift2 = copy.copy(fshift)

#on place un carré de 40 par 40 au centre = on filtre les hautes fréquences
fshift2[crow-200:crow+200, ccol-200:ccol+200] = 1


#On plot la transformée inverse logarithmiquement
magnitude_spectrum2 = np.log(np.abs(fshift2))

#On fait la transformée inverse
img_back2 = np.fft.ifft2(fshift2)
img_back2 = np.abs(img_back2)

#image de base
plt.subplot(151),plt.imshow(img2, cmap = 'gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])

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


#on tente un treshold sur les images filtrées
isize = img.shape

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        #laplacian[i][j] = abs(laplacian[i][j] - 255)
        sobelxy[i][j] = abs(sobelxy[i][j] - 255)
        #if sobelxyz[i][j] < 20 :
            #sobelxyz[i][j] = 0
        if laplacian[i][j] < 20 :
            laplacian[i][j] = 0


edges2 = cv2.Canny(img7,100,210)

cv2.imshow('lapla nega',laplacian)

kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow('lapla open close', closing)

cv2.imshow('edges2',edges2)

cv2.imshow('sobelxyz',sobelxyz)
'''

'''
#on peut également tenter de détecter les cercles sur l'image

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))


for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('circles',img)
'''

#on va utiliser un filtre de perwitt

#on créé le kernel pour le filtre sur x
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
#on créé le kernel pour le filtre sur y
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

#on applique le filtre en x
img_prewittx = cv2.filter2D(img, -1, kernelx)

#on applique le filtre en y
img_prewitty = cv2.filter2D(img, -1, kernely)

#on affiche les résultats
#cv2.imshow('prewitt y', img_prewitty)
#cv2.imshow('prewitt x', img_prewittx)

#n'étant pas convaincant tous seuls on les combine
prewitt = img_prewittx+img_prewitty

#et on affiche le résultat
cv2.imshow('prew',prewitt)

#histogramme de prewitt final
histo3 = plt.hist(prewitt.ravel(),256,[0,256],alpha=0.5,label="prewitt")

#normalisation
cv2.normalize(prewitt, prewitt, 255, 255, cv2.NORM_INF )

#on voit que la normalisation ne change rien car tout le spectre est utilisé
histo4 = plt.hist(prewitt.ravel(),256,[0,256],alpha=0.5,label="prewitt norma")

#on affiche le prewitt normalisé
#cv2.imshow('prewitt norma',prewitt)


#la normalisation ne fait rien mais on remarque qu'il n'y a presque rien au delà du 100
#on va donc changer la valeur des pixels au delà de 100 à 100
for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if prewitt[i][j] > 100:
            prewitt[i][j] =100

#on peut maintenant normaliser l'image
cv2.normalize(prewitt, prewitt, 255, 255, cv2.NORM_INF )

#on affiche le nouvel histogramme
histo4 = plt.hist(prewitt.ravel(),256,[0,256],alpha=0.5,label="prew norma 2")

cv2.imshow('prewitt renorma', prewitt)

#on effectue un thresholding sur l'image renormalisée
for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if prewitt[i][j] > 150:
            prewitt[i][j] = 255
        else :
            prewitt[i][j] = 0

cv2.imshow('prew norm tresh',prewitt)
plt.legend(loc='upper right')
plt.show()


'''

#pour essayer on applique aussi un filtre de Roberts


#on créé le kernel pour le filtre sur y
kernelRobx = np.array([[1,0],[0,-1]])

#on créé le kernel pour le filtre sur x
kernelRoby = np.array([[0,1],[-1,0]])


#on applique le filtre en x
img_Robx = cv2.filter2D(img, -1, kernelRobx)

#on applique le filtre en yl
img_Roby = cv2.filter2D(img, -1, kernelRoby)

img_Rob = img_Robx+img_Roby

Robhisto = plt.hist(img_Rob.ravel(),256,[0,256],alpha=0.5,label="orginale")

cv2.imshow('robx',img_Robx)
cv2.imshow('roby',img_Roby)
cv2.imshow('rob',img_Rob)

cv2.normalize(img_Rob, img_Rob, 255, 255, cv2.NORM_INF )

Robhisto = plt.hist(img_Rob.ravel(),256,[0,256],alpha=0.5,label="norm")

cv2.imshow('robNorm',img_Rob)

for i in range(0, isize[0]):
    for j in range(0, isize[1]):
        if img_Rob[i][j] > 35:
            img_Rob[i][j] = 255
        else :
            img_Rob[i][j] = 0

cv2.imshow('robNormTresh',img_Rob)

plt.show()
'''

cv2.waitKey(0)
cv2.destroyAllWindows()