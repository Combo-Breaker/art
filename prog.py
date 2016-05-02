import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage import data, io, measure, segmentation
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import misc
import copy
from skimage.future import graph
from skimage.util.colormap import viridis
from matplotlib import colors

# load the games image
#img = cv.imread("mem_persist_dali.jpg")

img = io.imread("mem_persist_dali.jpg")
img = cv.GaussianBlur(img, (11,11), 0)
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
labels = watershed(gradient, markers)
#labels = segmentation.slic(denoised, compactness=15, n_segments=300)
g = graph.rag_mean_color(img, labels)

cmap = colors.ListedColormap(['#6599FF', '#ff9900'])
out = graph.draw_rag(labels, g, img, colormap = cmap)


'''
plt.figure()
plt.title("RAG with all edges shown in green.")
plt.imshow(labels)
plt.show()



#array with size and perimeter of segmets
properties = measure.regionprops(labels)
#properties[i].area && properties[i].perimeter
print(len(properties))


'''
# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 12), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0, ax1 = axes

ax0.imshow(out, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Graph")
ax1.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax1.set_title("Segmented")
'''
ax2.imshow(out, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title("Graph")
ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.3)
ax3.set_title("Segmented")
'''
for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
