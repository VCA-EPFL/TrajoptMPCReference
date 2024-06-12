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



def runSolversSQP(trajoptMPCReference: TrajoptMPCReference, N: int, dt: float, solver_methods: List[SQPSolverMethods], options = {}):
	for solver in solver_methods:
		print("-----------------------------")
		print("Solving with method: ", solver)


		nq = trajoptMPCReference.plant.get_num_pos()
		nv = trajoptMPCReference.plant.get_num_vel()
		nx = nq + nv
		nu = trajoptMPCReference.plant.get_num_cntrl()
		if(options['overloading']):
			x = matrix_(np.zeros((nx,N)))
			u = matrix_(np.zeros((nu,N-1)))
		else:
			x = np.zeros((nx,N))
			u = np.zeros((nu,N-1))
		xs = copy.deepcopy(x[:,0])
		t1 = time.perf_counter(), time.process_time()
		x, u  = trajoptMPCReference.SQP(x, u, N, dt, LINEAR_SYSTEM_SOLVER_METHOD = solver, options = options)
		t2 = time.perf_counter(), time.process_time()

		# Save state for display
		# type_cost=sys.argv[1]
		type_cost='urdf'

		csv_file_path = f'data/{type_cost}/final_state.csv'
		with open(csv_file_path, 'w', newline='\n') as file:
			csv_writer = csv.writer(file)
			csv_writer.writerows(x)

		J = 0
		# Cost for last iteration/trajectory, sum over the horizon
		for k in range(N-1):
			J += trajoptMPCReference.cost.value(x[:,k], u[:,k])
			print("x: ", x[:,k])
		J += trajoptMPCReference.cost.value(x[:,N-1], None)

		print("Cost [", J, "]")
		print("Final State Error vs. Goal")
		error=0

		if isinstance(trajoptMPCReference.cost,UrdfCost):
			error=trajoptMPCReference.cost.delta_x(x[:,-1])

		if isinstance(trajoptMPCReference.cost,QuadraticCost):
			error=x[:,-1] - trajoptMPCReference.cost.xg
		if isinstance(trajoptMPCReference.cost,ArmCost):
			error=trajoptMPCReference.cost.current_state(x[:,-1])-trajoptMPCReference.cost.xg

	return error

	
		

def runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, solver_methods, options = {}):
	print("-----------------------------")
	print("-----------------------------")
	print("    Running SQP Example      ")
	print("-----------------------------")
	print("-----------------------------")
	print("Solving Unconstrained Problem")
	print("-----------------------------")

	
	trajoptMPCReference = TrajoptMPCReference(plant, cost)
	
	error=runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	if hard_constraints is not None:
		print("---------------------------------")
		print("---------------------------------")
		print(" Solving Constrained Problem Hard")
		print("---------------------------------")
		trajoptMPCReference = TrajoptMPCReference(plant, cost, hard_constraints)
		runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)
		
	if soft_constraints is not None:
		print("---------------------------------")
		print("---------------------------------")
		print(" Solving Constrained Problem Soft")
		print("---------------------------------")
		trajoptMPCReference = TrajoptMPCReference(plant, cost, soft_constraints)
		runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options)

	return error

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