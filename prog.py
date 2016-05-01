# import the necessary packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage import data, io
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import misc
import copy

# load the games image
#img = cv.imread("mem_persist_dali.jpg")
#img = cv.GaussianBlur(image, (11,11), 0)
img = io.imread("mem_persist_dali.jpg")
dst = copy.deepcopy(img) 

spatial_radius = 30
color_radius = 30
cv.pyrMeanShiftFiltering(img, spatial_radius, color_radius, dst) 

denoised = rank.median(dst[:, :, 0], disk(2))


# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))

# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Original")
#ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
#ax1.set_title("Local Gradient")
#ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
#ax2.set_title("Markers")
ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()

#cv.imshow("Image", img)
#cv.waitKey(0)





