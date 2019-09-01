import matplotlib.pyplot as plt 
import numpy as np 

n_pts = 100
random_x1_points = np.random.normal(10, 2, n_pts)
random_x2_points = np.random.normal(12, 2, n_pts)
#Find the transpose because we need to convert the two rows of data to have two columns of the form [x1, x2]
top_region = np.array([random_x1_points, random_x2_points]).T 
random_x1_points = np.random.normal(5, 2, n_pts)
random_x2_points = np.random.normal(6, 2, n_pts)
#Find the transpose because we need to convert the two rows of data to have two columns of the form [x1, x2]
bottom_region = np.array([random_x1_points, random_x2_points]).T 

#use subplots to have two plots in the same plot
_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], c='red')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], c='blue')
plt.show()