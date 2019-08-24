import cv2
import numpy as np 
import matplotlib.pyplot as plt 

#This code detects lanes in videos
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

Line detection using Hough Transform:
Every line can be represented in the form of y = mx + b in cartesian coordinates.
A point (x,y) can have infinite possible values for m and b since infinite number of lines
(family of lines) can pass through the point (x,y)

Hough Space is represented by a 2D plane with slope m as the X-axis and intercept b as Y-axis 
in the form of a line connecting m and b. 
If there are 2 points in a cartesian plane such that a line can be drawn through them, then it means
that in the Hough Space, their corresponding lines meet at a point.
Consider multiple points on a cartesian plane, if a line can be drawn through the points, it means that
in the Hough space there will be a point of intersection for some value of (m,b). 
When one line cannot be drawn through all the points, then we draw a grid in the hough space and select the 
cell where maximum number of lines in Hough Space intersect each other. We then draw a line through those points
only in the cartesian plane.

We need to use polar cordinates because perpendicular lines cannot be drawn in cartesian plane.

In polar coordinates,
P = xcos@ + ysin@
where @ is the angle made by the line with x-axis. 

Hough space consists of P on the y-axis and @ on x-axis. It has sinosodial wave forms and the point
(@,P) where the waves intersect is the value which will give a line through the points.


"""
def canny_image(image):
	#convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
	#Let us now use bitwise and operation between every pixel in the masked image and every
	#pixel in the canny image to get the actual ROI that we need
	masked_image = cv2.bitwise_and(mask, image)
	return masked_image

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			#lines is by default in 2D, so reshape it to 1D
			#unpack the end points of line
			x1, y1, x2, y2 = line.reshape(4)
			#draw the line on the black image
			#10 is line strength
			#(255, 0 ,0) means display blue line with 255 intensities and dont display red and green
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

	return line_image

def make_coordinates(image, line_fit):
	slope, intercept = line_fit
	#Let us use a line on either side that starts from bottom and goes till 3/5th of image
	y1 = image.shape[0]
	y2 = int(y1 * 3/5)
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	#lines on left side have negative slope and right side have positive slope.
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		#We fit a one degree polynomial curve (linear) and get the coefficient values
		#slope and intercept polynomial values are returned
		parameters = np.polyfit((x1,x2), (y1,y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope,intercept))
	#axis=0 implies averaging along the columns
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

cap = cv2.VideoCapture("Video/test2.mp4")
while (cap.isOpened()):
	#Read in the image 
	_, image = cap.read()
	#create a copy of image to prevent referencing
	image_copy = np.copy(image)
	canny = canny_image(image_copy)
	#matplotlib will show the image as well as a grid so we get to know the co-ordinates for ROI
	#Use matplotlib imshow only when co-ordinates are needed.
	# plt.imshow(canny)
	# plt.show()
	masked_image = region_of_interest(canny)
	#idntify all lines in the image
	"""
	Arguments for HoughLinesP function:
	1.image
	2. Pixel error which can be accepted while constructing grid
	3. Angle errors in radians which can be accepted
	4. Threshold (min number of intersections of lines)
	5.minLineLength: Minimum length of the line to consider it valid
	6. maxLineGap: Maximum gap that can be present between two segmented lines.
	"""
	lines = cv2.HoughLinesP(masked_image, 2 , np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=4)
	#instead of having multiple chopped lines, we can find out the averaged single line 
	averaged_lines = average_slope_intercept(image_copy, lines)
	line_image = display_lines(image_copy, averaged_lines)
	#Now add this detected line on top of the origin image to show the detected line
	#For this we add the identified lines on top of the original image. We add them and not bitwise AND
	#them because we are not dealing with black and white images.
	#We reduce the intensity of original image by multiplying it with 0.8 and increase the intensity of line image
	#set gamma to 1
	combo_image = cv2.addWeighted(image_copy, 0.8, line_image, 1, 1)
	cv2.imshow('result', combo_image)
	#waitKey can be used to wait for some period of time before a key is pressed. 0 means infinite waiting time
	#if user presses q key, let us quit
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

