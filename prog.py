import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage import data, io, measure, segmentation, color
from skimage.filters import rank
from skimage.draw import line, set_color
from skimage.util import img_as_ubyte
from scipy import misc
import copy
from skimage.future import graph
from skimage.util.colormap import viridis
from matplotlib import colors
import networkx as nx
from skimage.measure import regionprops
from skimage import draw


# load the games image
#img = cv.imread("mem_persist_dali.jpg")

img = io.imread("coins.jpg")
img = cv.GaussianBlur(img, (11,11), 0)
dst = copy.deepcopy(img) 

spatial_radius = 30
color_radius = 20
cv.pyrMeanShiftFiltering(img, spatial_radius, color_radius, dst) 

denoised = rank.median(dst[:, :, 0], disk(2))

'''
# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]
# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))
labels = watershed(gradient, markers)
'''

labels = segmentation.slic(denoised, compactness=1, n_segments=40)
print(labels)

#markers = rank.gradient(denoised, disk(5)) < 10
#markers = ndi.label(markers)[0]
# local gradient (disk(2) is used to keep edges thin)
#gradient = rank.gradient(denoised, disk(2))
#labels = watershed(gradient, markers)

'''
s = graph.rag_mean_color(img, labels)
tr = 
g = graph.merge_hierarchical(labels, s, tr)
cmap = colors.ListedColormap(['blue', 'red'])
out = graph.draw_rag(labels, g, img, colormap=cmap)
#print(g.nodes())
'''
'''
rr, cc = line (585, 587, 516, 518)
set_color(labels, (rr, cc), 0)
print(ndi.measurements.center_of_mass(labels))
'''
#plt.imshow(out)
#plt.show()

'''
#array with size and perimeter of segmets
properties = measure.regionprops(labels)
#properties[i].area && properties[i].perimeter
print(len(properties))



# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 12), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
ax0 = axes

ax0.imshow(out, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title("Graph")

ax1.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax1.set_title("Segmented")

ax2.imshow(out, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title("Graph")
ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.3)
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
'''
