import cv2
import copy
import imutils

img = cv2.imread('pics/U2_surface.pbm')
img4 = copy.copy(img)

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

'''
canny effectue:
    une réduction du bruit avec un filtre gaussian de 5x5
    un filtre de soel de sobel horizontal et vertical, ce qui lui permet d'obtenir les première dérivée
    avec ces dérivées il calcule l' edge gradient ( sqrt(deriv_X² + deriv_Y²)) et la direction du gradient ( artan(deriv_Y/deriv_X) <- arondit à 45°
    une opération pour réduire l'épaisseur des bords
        on place un point A sur le bord, deux points B et C le long de la direction du gradient, si A forme un maximum local on le garde snion on le met à 0
        un thresholding 
'''
canny = cv2.Canny(img, 255,255)

'''
findcontour trouve les contours dans une image binaire, l'algorithme utilisé est derrière un paywall
'''
contour = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #RETR_TREE récupère tous les contours et les tires, CHAIN_APPROX_SIMPLE compresse les contours en leurs extrémitées

# sans ça "contours is not a numpy array, neither a scalar"
img3 = contour[0] if imutils.is_cv2() else contour[1]

#on trie les contours trouvés pour avoir le plus grand au début
img3 = sorted(img3, key = cv2.contourArea, reverse = True)[:1]

#on trace un traite bleu de 2 pixels de large autour du contour, ergo autour de l'objet
cv2.drawContours(img4,img3,0,(255,0,0),2)

#on affiche le canny sur lequel on peut clairement identfier l'objet
cv2.imshow('canny',canny)

#on affiche l'iamge avec le TARDIS entouré
cv2.imshow('entouré',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()