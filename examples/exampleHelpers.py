import csv

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from TrajoptPlant import TrajoptPlant, DoubleIntegratorPlant, PendulumPlant, CartPolePlant, URDFPlant
from TrajoptCost import TrajoptCost, QuadraticCost, ArmCost, UrdfCost, NumericalCost
from TrajoptConstraint import TrajoptConstraint, BoxConstraint
from TrajoptMPCReference import TrajoptMPCReference, SQPSolverMethods, MPCSolverMethods

import numpy as np
import copy
from typing import List
import time
from overloading import matrix_

i=0

def joint_angles_to_ee_pos(l1:float,l2:float,state: np.ndarray):


	posx=l2*np.cos(state[0],state[1])+ l1*np.cos(state[0])
	posy=l2*np.sin(state[0],state[1])+ l1*np.sin(state[0])

	return posx,posy


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



def runSolversSQP(trajoptMPCReference: TrajoptMPCReference, N: int, dt: float, solver_methods: List[SQPSolverMethods], options = {}):
	global i
	for solver in solver_methods:
		print("-----------------------------")
		print("Solving with method: ", solver)


		nq = trajoptMPCReference.plant.get_num_pos()
		nv = trajoptMPCReference.plant.get_num_vel()
		nx = nq + nv
		nu = trajoptMPCReference.plant.get_num_cntrl()
		# x = matrix_(np.zeros((nx,N)))
		# u = matrix_(np.zeros((nu,N-1)))
		x = np.zeros((nx,N))
		u = np.zeros((nu,N-1))
		xs = copy.deepcopy(x[:,0])
		t1 = time.perf_counter()#, time.process_time()
		x, u = trajoptMPCReference.SQP(x, u, N, dt, LINEAR_SYSTEM_SOLVER_METHOD = solver, options = options)
		t2 = time.perf_counter()#, time.process_time()

		# Save state for display
		type_cost=sys.argv[1]
		csv_file_path = f'data/multiple_test/final_state_{i}.csv'
		with open(csv_file_path, 'w', newline='\n') as file:
			csv_writer = csv.writer(file)
			csv_writer.writerows(x)

		# if options["display"]:
		# 	display(x, title="SQP Solver Method: " + solver.name)

		J = 0
		# Cost for last iteration/trajectory, sum over the horizon
		for k in range(N-1):
			J += trajoptMPCReference.cost.value(x[:,k], u[:,k])
			print("x: ", x[:,k])
			#saved_J.append([trajoptMPCReference.cost.value(x[:,k], u[:,k])])
		J += trajoptMPCReference.cost.value(x[:,N-1], None)
		#saved_J.append([trajoptMPCReference.cost.value(x[:,N-1], None)])

		print("Cost [", J, "]")
		print("Final State Error vs. Goal")


		if isinstance(trajoptMPCReference.cost,UrdfCost):
			error=trajoptMPCReference.cost.delta_x(x[:,-1])

		if isinstance(trajoptMPCReference.cost,QuadraticCost):
			error=x[:,-1] - trajoptMPCReference.cost.xg

		
		i=i+1
		run_time=t2-t1

		return run_time, error
		# csv_file_path = f'data/{type_cost}/J.csv'
		# with open(csv_file_path, 'w', newline='\n') as file:
		# 	csv_writer = csv.writer(file)
		# 	csv_writer.writerows(saved_J)


	
		

def runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, solver_methods, options = {}):
	print("-----------------------------")
	print("-----------------------------")
	print("    Running SQP Example      ")
	print("-----------------------------")
	print("-----------------------------")
	print("Solving Unconstrained Problem")
	print("-----------------------------")

	
	trajoptMPCReference = TrajoptMPCReference(plant, cost)
	
	run_time, error= runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)


	# print("---------------------------------")
	# print("---------------------------------")
	# print(" Solving Constrained Problem Hard")
	# print("---------------------------------")
	# trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
	# runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	# print("---------------------------------")
	# print("---------------------------------")
	# print(" Solving Constrained Problem Soft")
	# print("---------------------------------")
	# trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
	# runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	return run_time, error
	

def runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options = {}):
	for solver in solver_methods:
		print("-----------------------------")
		print("Solving with method: ", solver)

		nq = trajoptMPCReference.plant.get_num_pos()
		nv = trajoptMPCReference.plant.get_num_vel()
		nx = nq + nv
		nu = trajoptMPCReference.plant.get_num_cntrl()
		x = np.zeros((nx,N))
		u = np.zeros((nu,N-1))
		xs = copy.deepcopy(x[:,0])

		print("Goal Position")
		print(trajoptMPCReference.cost.xg)
		print("-------------")

		t1 = time.perf_counter(), time.process_time()
		x, u = trajoptMPCReference.MPC(x, u, N, dt, SOLVER_METHOD = solver)
		t2 = time.perf_counter(), time.process_time()

		if options["display"]:
			display(x, title="MPC Solver Method: " + solver.name)

		if isinstance(trajoptMPCReference.cost,QuadraticCost):
			error=x[:,-1] - trajoptMPCReference.cost.xg
			
		if isinstance(trajoptMPCReference.cost,ArmCost):
			state, _ =trajoptMPCReference.cost.symbolic_cost_eval()
			error=np.array(state(*x[:,-1]))-np.array(trajoptMPCReference.cost.xg).reshape(4,1)

		return t2[0]-t1[0], t2[1] - t1[1], np.linalg.norm(error[:2])


def runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, solver_methods, options = {}):
	print("-----------------------------")
	print("-----------------------------")
	print("    Running MPC Example      ")
	print("-----------------------------")
	print("-----------------------------")
	print("Solving Unconstrained Problem")
	print("-----------------------------")
	trajoptMPCReference = TrajoptMPCReference(plant, cost)
	run_time, cpu_time, error=runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options)

	# print("---------------------------------")
	# print("---------------------------------")
	# print(" Solving Constrained Problem Hard")
	# print("---------------------------------")
	# trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
	# runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options)

	# print("---------------------------------")
	# print("---------------------------------")
	# print(" Solving Constrained Problem Soft")
	# print("---------------------------------")
	# trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
	# runSolversMPC(trajoptMPCReference, N, dt, solver_methods, options)

	return run_time, cpu_time, error