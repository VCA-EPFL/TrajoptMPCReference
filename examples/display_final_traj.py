import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def display(x: np.ndarray, x_lim: List[float] = [-2.5, 2.5], y_lim: List[float] = [-2.5, 2.5], title: str = ""):
	print("display")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	line1, = ax.plot([0, 0, 0], [0, 1, 2], 'b-')
	ax.set_xlim(x_lim)
	ax.set_ylim(y_lim)
	fig.suptitle(title)
	N = x.shape[1]
	type_cost=sys.argv[1]
	for k in range(N):
		print("State at time step ", k, " is: ", x[:,k])
		first_point = [0, 0]
		second_point = [-np.sin(x[0,k]), np.cos(x[0,k])]
		third_point = [second_point[0] - np.sin(x[0,k]+x[1,k]), second_point[1] + np.cos(x[0,k]+x[1,k])]
		line1.set_xdata([first_point[0], second_point[0], third_point[0]])
		line1.set_ydata([first_point[1], second_point[1], third_point[1]])
		plt.title("Time Step: " + str(k))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.1)
		plt.savefig(f'results/{type_cost}/state')


# Read state from file
type_cost=sys.argv[1]
x = np.loadtxt(f'data/{type_cost}/final_state.csv', delimiter=',')

display(x, title="SQP Solver Method:{type_cost} cost")

