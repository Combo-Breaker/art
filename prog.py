# import the necessary packages
import numpy as np
import cv2 as cv
 
# load the games image
img = cv.imread("mem_persist_dali.jpg")
#img = cv.GaussianBlur(image, (11,11), 0)
dst = img

spatial_radius = 30
color_radius = 30
cv.pyrMeanShiftFiltering(img, spatial_radius, color_radius, dst) 


cv.imshow("Image", dst)
cv.waitKey(0)





