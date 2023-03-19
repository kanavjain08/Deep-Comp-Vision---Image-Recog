import numpy as np
import cv2 as cv

#image
image = cv.imread('Photos/wallpaper.jpg')
#cv.imshow('image', image)

#drawing on the image
blank = np.zeros(image.shape[:2], dtype = 'unit8')
cv.imshow('blank', blank)

#grayscale image
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('grey', grayscale)

#blurring the image
#blurred = cv.GaussianBlur(grayscale, (5,5), cv.BORDER_DEFAULT)
#cv.imshow('blur', blurred)

#number of edges
#edgedetector = cv.Canny(blurred, 125, 175)
#cv.imshow('canny', edgedetector)
 
#binarisez image, below 125 -> 0, above -> 1 (black/white)
ret, thresh = cv.threshold(grayscale, 125, 255, cv.THRESH_BINARY)
cv.imshow('thresh', thresh)

#returns number of contours and hierarchies
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')


cv.waitKey(0)