from __future__ import division
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
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

def dist(x, y):
	return (np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))

def D(seg1, seg2, img): #from seg1 to all others
	height, width, channels = img.shape
	image_area = height*width
	S = ((seg2.area)/image_area)
	distance = dist(seg1.centroid, seg2.centroid)
	res = 1 - (distance*S)/np.sqrt(height**2 + width**2)
	return(res)


#img = io.imread("Raphael_Galatea.jpg")
img = io.imread("sportsmen.jpg")
#img = io.imread("interior.jpg")



img = cv.GaussianBlur(img, (11,11), 0)

labels = segmentation.slic(img, compactness=10, n_segments=400) 
#compactness depends on the level of contrast Galatea (10). Sportsmen(30)
labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
regions = regionprops(labels) 
# Returns  region properties such as area (number of pixels of region), centroid (centroid coordinate tuple [row, col]),



reg_distances = [[0 for x in range(2)] for y in range(len(regions))] 
i = -1
for r in regions:
	s = 0
	i += 1
	for t in regions:
		if (t != r):
			s += D(r, t, img)
			reg_distances[i][0] = s
			x = int(t.centroid[0])
			y = int(t.centroid[1])
			color = img[x, y]
			reg_distances[i][1] = color

'''
print(reg_distances[1:6])
print(reg_distances[100:106])
'''


label_rgb = color.label2rgb(labels, img, kind='avg')

label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))

rag = graph.rag_mean_color(img, labels)


for region in regions:
    rag.node[region['label']]['centroid'] = region['centroid']

def display_edges(image, g,):
    """Draw edges of a RAG on its image

    Returns a modified image with the edges drawn.Edges are drawn in green
    and nodes are drawn in yellow.

    Parameters
    ----------
    image : ndarray
        The image to be drawn on.
    g : RAG
        The Region Adjacency Graph.
    threshold : float
        Only edges in `g` below `threshold` are drawn.

    Returns:
    out: ndarray
        Image with the edges drawn.
    """
    image = image.copy()
    for edge in g.edges_iter():
    	n1, n2 = edge

        r1, c1 = map(int, rag.node[n1]['centroid'])
        r2, c2 = map(int, rag.node[n2]['centroid'])

        line  = draw.line(r1, c1, r2, c2)
        circle = draw.circle(r1,c1,2)

        #if g[n1][n2]['weight'] < high and g[n1][n2]['weight'] > low:
        weight_int = g.node[n1]['mean color'].astype(int) - g.node[n2]['mean color'].astype(int)
        weight_int = np.linalg.norm(weight_int)
        weight_double = g[n1][n2]['weight']

    	if weight_int > 30 and weight_double < 30 :
            print ("Double Vectors")
            print ("Vector 1",g.node[n1]['mean color'])
            print ("Vector 2",g.node[n2]['mean color'])
            print ("Difference",g.node[n1]['mean color'] - g.node[n2]['mean color'])
            print ("Magnitude",weight_double)
            print ("Int Vectors")
            print ("Vector 1",g.node[n1]['mean color'].astype(int))
            print ("Vector 2",g.node[n2]['mean color'].astype(int))
            print ("Difference",g.node[n1]['mean color'].astype(int) - g.node[n2]['mean color'].astype(int))
            print ("Magnitude",weight_int)

            image[line] = 0,1,0

        image[circle] = 1,1,0

    return image

def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)

px = img[700, 3]
print("COLOR ", px)
edges_drawn = display_edges(label_rgb, rag)
show_img(edges_drawn)
plt.figure(figsize = (12,8))
plt.show()

