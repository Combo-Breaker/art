from __future__ import division
import numpy as np
import cv2 as cv
import matplotlib
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
import struct
from os import listdir


def similar_colors(c1, c2, tr): #color1, color2, treshold
	if np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2) < tr :
		return True
	else:
		return False

def to_hsv(rgb):
	maximum = max(rgb[0], rgb[1], rgb[2])/255
	minimum = min(rgb[0], rgb[1], rgb[2])/255
	if maximum == 0:
		s = 0
	else:
		s = 1 - minimum/maximum
	res = [s, maximum]
	return res


files = listdir("pic/Renaissance")
for l in files:
	s = "pic/Renaissance/"
	s += str(l)
	print(s)
	img = io.imread(s)
	img = cv.GaussianBlur(img, (11,11), 0)

	labels = segmentation.slic(img, compactness=10, n_segments=200) 
	#compactness depends on the level of contrast Galatea (10). Sportsmen(30)
	labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
	regions = regionprops(labels) 
	# Returns  region properties such as area (number of pixels of region), centroid (centroid coordinate tuple [row, col]),

	'''
	palette = []
	palette.append(img[regions[0].centroid[0], regions[0].centroid[1]])

	#getting the image palette
	for j in range (len(regions)):
		reg = regions[j]
		coords = reg.centroid
		clr = img[coords[0], coords[1]]
		i = 0	
		for c in palette:
			if similar_colors(c, clr, 30):
				i += 1
				break
		if i == 0:
			palette.append(clr)
	'''

	label_rgb = color.label2rgb(labels, img, kind='avg')
	label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
	rag = graph.rag_mean_color(img, labels)
	for region in regions:
	    rag.node[region['label']]['centroid'] = region['centroid']
	
	#the left side is more calm
	height, width, channels = img.shape
	image_area = height*width
	left_saturation = []
	left_brightness = []
	right_saturation = []
	right_brightness = []
	left_s = 0
	left_b = 0
	right_s = 0
	right_b = 0
	for reg in regions:
		coords = reg.centroid
		s = ((reg.area)/image_area)
		if (coords[1] <= width/2):
			left_s  += s * (to_hsv(img[coords[0], coords[1]])[0])
			left_b += s * (to_hsv(img[coords[0], coords[1]])[1])
		else:
			right_s  += s * (to_hsv(img[coords[0], coords[1]])[0])
			right_b += s * (to_hsv(img[coords[0], coords[1]])[1])

	#print("saturation: ", left_saturation*100, right_saturation*100)
	#print("brightness: ", left_brightness*100, right_brightness*100)
	left_saturation.append(left_s*100)
	left_brightness.append(left_b*100)
	right_brightness.append(right_b*100)
	right_saturation.append(right_s*100)


	'''
	def display_edges(image, g,):
	    image = image.copy()
	    for edge in g.edges_iter():
	    	n1, n2 = edge
	        r1, c1 = map(int, rag.node[n1]['centroid'])
	        r2, c2 = map(int, rag.node[n2]['centroid'])
	        line  = draw.line(r1, c1, r2, c2)
	        circle = draw.circle(r1,c1,2)
	        weight_int = g.node[n1]['mean color'].astype(int) - g.node[n2]['mean color'].astype(int)
	        weight_int = np.linalg.norm(weight_int)
	        weight_double = g[n1][n2]['weight']
	    	if weight_int > 30 and weight_double < 30 :            
	            image[line] = 0,1,0
	        image[circle] = 1,1,0
	    return image
	
	def show_img(img):
	    width = 10.0
	    height = img.shape[0]*width/img.shape[1]
	    f = plt.figure(figsize=(width, height))
	    plt.imshow(img)
	    plt.show()

	#px = img[700, 3]
	#print("COLOR ", px)
	edges_drawn = display_edges(label_rgb, rag)
	show_img(edges_drawn)
	

	#draw the palette
	fig = plt.figure()
	ax = fig.add_subplot(111)
	i = 0
	j = 0
	rect = []
	for c in palette:
		rect.append(matplotlib.patches.Rectangle((i,j), 70, 70, color=to_hex(c)))
		i += 70
		if (i >= 600):
			i = 0
			j += 70
		
	for r in rect:
		ax.add_patch(r)
	plt.xlim([0, 1000])
	plt.ylim([0, 1000])
	plt.show()
	'''
print("rs", right_saturation, "rb", right_brightness)
print("ls", left_saturation, "lb", left_brightness)