import sys
import csv
import numpy as np

# Read state from file
type_cost=sys.argv[1]
x = np.loadtxt(f'data/{type_cost}/final_state.csv', delimiter=',')

display(x, title="SQP Solver Method:{type_cost} cost")

def display(x: np.ndarray, x_lim: List[float] = [-2.5, 2.5], y_lim: List[float] = [-2.5, 2.5], title: str = ""):
	print("display")
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# line1, = ax.plot([0, 5, 10], [0, 5, 10], 'b-')
	line1, = ax.plot([0, 0, 0], [0, 1, 2], 'b-')
	ax.set_xlim(x_lim)
	ax.set_ylim(y_lim)
	# set suptitle as title
	fig.suptitle(title)
	N = x.shape[1]
	type_cost=sys.argv[1]
	for k in range(N):
		print("State at time step ", k, " is: ", x[:,k])
		# if type_cost in ['urdf','sym']:
		# 	x=-np.sin(x[0,k]+x[1,k])-np.sin(x[0,k])
		# 	y=np.cos(x[0,k]+x[1,k])+np.cos(x[0,k])
		# 	print("End effector postition at time step ", k, " is x: ", x, ", y: ", y)

		# x[:,k] is the state at time step k
		# the first number is the angle of the first joint
		# the second number is the angle of the second joint
		# draw the line with a length of 5
		# add 90 degrees to the angle to make it point up
		first_point = [0, 0]
		second_point = [-np.sin(x[0,k]), np.cos(x[0,k])]
		third_point = [second_point[0] - np.sin(x[0,k]+x[1,k]), second_point[1] + np.cos(x[0,k]+x[1,k])]
		line1.set_xdata([first_point[0], second_point[0], third_point[0]])
		line1.set_ydata([first_point[1], second_point[1], third_point[1]])
		plt.title("Time Step: " + str(k))
		fig.canvas.draw()
		#fig.canvas.mpl_connect('close_event', _on_close)
		fig.canvas.flush_events()
		plt.pause(0.1)

	plt.show()