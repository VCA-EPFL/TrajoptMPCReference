import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List

#n_joints=sys.argv[1]
# def display(x: np.ndarray, x_lim: List[float] = [-2.5, 2.5], y_lim: List[float] = [-2.5, 2.5], title: str = ""):
def display(x: np.ndarray, x_lim: List[float] = [-3.5, 3.5], y_lim: List[float] = [-3.5, 3.5], title: str = ""):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	line1, = ax.plot([0, 0, 0, 0], [0, 1, 2, 3], 'b-')
	ax.set_xlim(x_lim)
	ax.set_ylim(y_lim)
	ax.grid(True)
	fig.suptitle(title)
	ax.set_xticks(np.arange(x_lim[0], x_lim[1] + 0.5, 0.5))
	ax.set_yticks(np.arange(y_lim[0], y_lim[1] + 0.5, 0.5))
	N = x.shape[1]
	for k in range(N):
		#print("State at time step ", k, " is: ", x[:,k])
		#generalize for n points and with different ls
		first_point = [0, 0]
		second_point = [-np.sin(x[0,k]), np.cos(x[0,k])]
		third_point = [second_point[0] - np.sin(x[0,k]+x[1,k]), second_point[1] + np.cos(x[0,k]+x[1,k])]
		#fourth_point = [third_point[0] - np.sin(x[0,k]+x[1,k]+x[1,k]), third_point[1] + np.cos(x[0,k]+x[1,k]+x[2,k])]
		line1.set_xdata([first_point[0], second_point[0], third_point[0]])#, fourth_point[0] ])
		line1.set_ydata([first_point[1], second_point[1], third_point[1]])#, fourth_point[1]])
		plt.title("Time Step: " + str(k))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.1)
		plt.savefig(f'results/urdf/state')


# Read state from file
# type_cost=sys.argv[1]
x = np.loadtxt(f'data/urdf/final_state.csv', delimiter=',')

display(x, title="SQP Solver Method:urdf cost")

