import matplotlib.pyplot as plt 
import numpy as np 

def draw(x1, x2):
	plt.plot(x1, x2)
	plt.show()

def sigmoid(score):
	return (1 + 1/np.exp(-score))

#any line is ofthe form w1x1 + w2x2 + b = 0, where b is bias and x1, x2 are the input variables
n_pts = 100
bias = np.ones(n_pts)
random_x1_points = np.random.normal(10, 2, n_pts)
random_x2_points = np.random.normal(12, 2, n_pts)
#Find the transpose because we need to convert the two rows of data to have two columns of the form [x1, x2]
top_region = np.array([random_x1_points, random_x2_points, bias]).T 
random_x1_points = np.random.normal(5, 2, n_pts)
random_x2_points = np.random.normal(6, 2, n_pts)
#Find the transpose because we need to convert the two rows of data to have two columns of the form [x1, x2]
bottom_region = np.array([random_x1_points, random_x2_points, bias]).T 
#let us initialize some weights
w1 = -0.2
w2 = -0.35
b = 3.5
#Find transpose for executing matrix multiplication
line_parameters = np.matrix([w1, w2, b]).T
#Select the point on the fat left and the point on the far right for x1
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -b/w2 + x1 * (-w1 / w2)
allpoints = np.vstack((top_region, bottom_region))
linear_combination = allpoints * line_parameters
print(linear_combination)
#Lets add a sigmoid function to the lienar_combination values to get the probability
score = sigmoid(linear_combination)
#use subplots to have two plots in the same plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], c='red')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], c='blue')
draw(x1, x2)
plt.show()