import cv2
import numpy as np 
import matplotlib.pyplot as plt 

"""
Steps to find lane lines:

1. Read the image 
2. Convert the image into grayscale so that it becomes easy for edge detection. Grayscale converts an RGB image to a
black and white sort of image where the image array consists of pixel intensities
ranging from 0 to 255. 0 is completely black (no intensity) and 255 is completely white. 

Important terms:
Edge: Edge is the drastic change in intensity of adjacent pixel values. For example, if a black image pixel 
suddenly changes into a white image pixel in the adjacent column or row, then we call it an edge.
Gradient: Measure of change of pixel intensity for adjacent pixels.
Stong gradient: Very high change in pixel intensity. Example: from 0 to 255.
Small gradient: Very low change in pixel intensity. Example: from 0 to 20.
To measure the gradient we find the derivative(f(x,y)), where
f(x,y) = Function that compares the intensity values of pixels horizontally (x) and vertically(y).
derivative(f(x,y)) = Measure of rate of change of pixel intensities horizontally and vertically.

3. Apply Gaussian blur in order to smoothen the image and remove the possibilty of finding
false edges in the images. For smoothening, we select a grid of some size (preferably (5,5)).
We traverse the image using this grid and calculate the average value in the grid and 
construct another image array which will be the blurred image. 
4. Apply Canny edge detection to detect the edges in the images. 
For canny edge detection, we decide upon the low and high thresholds for pixel intensities.
Any pixel with intensity more than high threshold is accepted and any pixel with intensity
less than low_threshold is rejected. If a pixel falls in between the low and high thresholds
then it will be selected if it is a main Edge. 
5. Display the image.

"""
def canny_image(image):
	#convert to grayscale
	gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny 

def region_of_interest(image):
	"""
	Finding the ROI involves finding the region of image \
	that we are interested in. In our case, we need to find \
	the traingle that is on the right side of the road \
	it starts from the middle of the road till the end of the road,
	and the tip of the traingle lies in the far off horizon.
	"""
	height = image.shape[0]
	#We need to create an array of polygons which wil be inside the traingle
	#Arguments are co-ordinates of left point, right point and midpoint of traingle.
	triangle = np.array([[(200, height), (1000, height), (550, 250)]])
	#Create a black mask image with same size as our canny/base image.
	mask = np.zeros_like(image)
	#Overlap our triangle on the mask image to get the ROI in white color.
	cv2.fillPoly(mask, triangle, 255)
	return mask 

#Read in the image 
image = cv2.imread('Image/test_image.jpg')
#create a copy of image to prevent referencing
image_copy = np.copy(image)
canny = canny_image(image_copy)
#matplotlib will show the image as well as a grid so we get to know the co-ordinates for ROI
#Use matplotlib imshow only when co-ordinates are needed.
# plt.imshow(canny)
# plt.show()
cv2.imshow('result', region_of_interest(canny))
#waitKey can be used to wait for some period of time before a key is pressed. 0 means infinite waiting time
cv2.waitKey(0)