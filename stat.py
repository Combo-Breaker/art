from __future__ import division
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage import data, io, measure, segmentation, color, filters
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
import struct
from os import listdir


def dist(x, y):
	return (np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))

def Distance(seg1, seg2, img): #from seg1 to others
	height, width, channels = img.shape
	image_area = height*width
	S = ((seg2.area)/image_area)
	distance = dist(seg1.centroid, seg2.centroid)
	res = 1 - (distance*S)/np.sqrt(height**2 + width**2)
	return(res) 

def similar_colors(c1, c2, tr): #color1, color2, treshold
	if (c1[0] - c2[0])**2 < tr and  (c1[1] - c2[1])**2 < tr and  (c1[2] - c2[2])**2 < tr :
		return True
	else:
		return False


files = listdir("pic/raphael")
for l in files:
	s = "pic/raphael/"
	s += str(l)
	print(s)
	img = io.imread(s)

	img = cv.GaussianBlur(img, (11,11), 0)
	labels = segmentation.slic(img, compactness=30, n_segments=150)
 	labels = labels + 1
 	regions = regionprops(labels)
 	edge_map = filters.sobel(color.rgb2gray(img))
 	label_rgb = color.label2rgb(labels, img, kind='avg')
 	label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 1, 1))
 	edge_map = filters.sobel(color.rgb2gray(img))
 	rag = graph.rag_boundary(labels, edge_map)
 	for region in regions:
 		rag.node[region['label']]['centroid'] = region['centroid']
 		rag.node[region['label']]['color'] = label_rgb[region['centroid']]

 	red = np.asarray([1, 0, 0])
	green = np.asarray([0, 1, 0])	
	blue = np.asarray([0, 0, 1])
	yellow = np.asarray([1, 1, 0])
	purple = np.asarray([0.5, 0, 0.5])
	orange = np.asarray([1, 0.5, 0])
	palette = [red, green, blue, yellow, purple, orange]

	harmony = 0
	disharmony = 0
	tr = 0.4
	edges = 0
	for edge in rag.edges_iter():
		edge_1, edge_2 = edge
		c1 = [float(elem) for elem in rag.node[edge_1]['color'].tolist()]
		c2 = [float(elem) for elem in rag.node[edge_2]['color'].tolist()]
		if (similar_colors(c1, red, tr)) and (similar_colors(c2, green, tr)) or \
			(similar_colors(c1, green, tr)) and (similar_colors(c2, red, tr)) or \
				(similar_colors(c1, yellow, tr)) and (similar_colors(c2, purple, tr)) or \
					(similar_colors(c1, purple, tr)) and (similar_colors(c2, yellow, tr)) or \
						(similar_colors(c1, blue, tr)) and (similar_colors(c2, orange, tr)) or \
							(similar_colors(c1, orange, tr)) and (similar_colors(c2, blue, tr)):
			harmony += 1
		else:
			flag1 = 0
			flag2 = 0
			for clr in palette:
				if similar_colors(clr, c1, tr):
					flag1 = 1
				if similar_colors(clr, c2, tr):
					flag2 = 1
			if (flag1 and flag2):
				disharmony += 1	

	    
	    

	print(harmony)
	print(disharmony)



	

    	


