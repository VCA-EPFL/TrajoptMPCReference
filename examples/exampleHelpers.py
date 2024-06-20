import csv

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from TrajoptPlant import TrajoptPlant,URDFPlant
from TrajoptCost import TrajoptCost, QuadraticCost, ArmCost, UrdfCost
from TrajoptConstraint import TrajoptConstraint, BoxConstraint
from TrajoptMPCReference import TrajoptMPCReference, SQPSolverMethods

import numpy as np
import copy
from typing import List
import time
from overloading import matrix_
import pandas as pd

# def save_in_file(file_path, value, overwrite):
# 	if "saved_" in file_path:
# 		file_path = file_path.replace("saved_", "")
# 	directory = os.path.dirname(file_path)
# 	if not os.path.exists(directory):
# 		os.makedirs(directory)
		
# 	if isinstance(value, float):
# 		value = [[value]]
# 	elif isinstance(value, list) and all(isinstance(i, float) for i in value):
# 		value = [[v] for v in value]
	
# 	if overwrite:
# 		mode='w'
# 	else: 
# 		mode= 'a'
# 	with open(file_path, mode, newline='\n') as file:
# 		csv_writer = csv.writer(file)
# 		try:
# 			csv_writer.writerows(value)
# 		except:
# 			print("error when saving in file: ", file_path)

def save_in_file(file_path, value, csv=False):

	if "saved_" in file_path:
		file_path = file_path.replace("saved_", "")

	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

	df = pd.DataFrame(value)
	if csv:
		df.to_csv(file_path)
	else:
		df.to_pickle(file_path)
		




def runSolversSQP(trajoptMPCReference: TrajoptMPCReference, N: int, dt: float, solver_methods: List[SQPSolverMethods], options = {},n_test=0, record=False):
	for solver in solver_methods:
		print("-----------------------------")
		print("Running test number: ", n_test)


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
		# t1 = time.perf_counter(), time.process_time()
		t1 = time.time()
		x, u, exit_sqp, exit_soft, outer_iter, sqp_iter   = trajoptMPCReference.SQP(x, u, N, dt, LINEAR_SYSTEM_SOLVER_METHOD = solver, options = options)
		# t2 = time.perf_counter(), time.process_time()
		t2 = time.time()

		
		if record:

			comp_time=t2-t1

			# Save final traj and input
			file_path_x = f'../data/{n_test}/final_traj.csv'
			save_in_file(file_path_x, x, csv=True)

			# with open(file_path_x,'w', newline='\n') as file:
			# 	csv_writer = csv.writer(file)
			# 	csv_writer.writerows(x)

			file_path_u = f'../data/{n_test}/final_input.csv'
			save_in_file(file_path_u, u, csv=True)
			
			
			J = 0
			Jx=0
			Ju=0
			# Cost for last iteration/trajectory, sum over the horizon
			for k in range(N-1):
				J += trajoptMPCReference.cost.value(x[:,k], u[:,k])
				# State dependent part of the cost => represents the error of the end effector position for the final trajectory
				Jx += trajoptMPCReference.cost.value(x[:,k], None)
				Ju += trajoptMPCReference.cost.value(x[:,k], u[:,k])- trajoptMPCReference.cost.value(x[:,k], None)
			J += trajoptMPCReference.cost.value(x[:,N-1], None)
			Jx += trajoptMPCReference.cost.value(x[:,N-1], None)

			
			# Input dependent part of the cost => represents energy expanditure for the final trajectory
			# Ju = J-Jx

			#Compute the Energy expanditure for the last trajectory
			# Power = torque * angular velocity, summed over each joint
			# Energy = Power *t 
			E = np.dot(u[:,-1],x[nq:,-1])*dt * 10000

			# Compute the error
			error=0
			if isinstance(trajoptMPCReference.cost,UrdfCost):
				error=trajoptMPCReference.cost.delta_x(x[:,-1])
			if isinstance(trajoptMPCReference.cost,QuadraticCost):
				error=x[:,-1] - trajoptMPCReference.cost.xg
			if isinstance(trajoptMPCReference.cost,ArmCost):
				error=trajoptMPCReference.cost.current_state(x[:,-1])-trajoptMPCReference.cost.xg

			results=[comp_time, J, Jx, Ju, error, E, exit_sqp, exit_soft,outer_iter, sqp_iter]
			results_file_path = f'../data/{n_test}/results.plk'
			save_in_file(results_file_path,results)
			
			# Save internal variables: SQP variables, Cost variables, Plant variables
			sqp_vars = ["saved_Pinv",\
						"saved_inner_traces","saved_G","trace","saved_invG","saved_c","saved_g","saved_C","saved_tot_cost",\
						"saved_J_tot_constraints","saved_jacobian_hard_constraints","saved_jacobian_soft_constraints","saved_Ak",\
						"saved_Bk","saved_xkp1","saved_dxul","saved_x","saved_u"]
			cost_vars=["saved_cost","saved_grad","saved_hess","saved_Jacobian_tot_state", "saved_dx"]
			plant_vars=["saved_Minv","saved_c", "saved_dc_du", "saved_qdd","saved_dqdd"]
			internal_vars=sqp_vars+ plant_vars + cost_vars


			for var in internal_vars:
				if var in sqp_vars:
					variable_to_save=getattr(trajoptMPCReference, var)
				elif var in cost_vars:
					variable_to_save=getattr(trajoptMPCReference.cost, var)
				elif var in plant_vars:
					variable_to_save=getattr(trajoptMPCReference.plant, var)

				file_path = f'../data/{n_test}/{var}.plk'
				save_in_file(file_path, variable_to_save)

			if options['overloading']:
				file_path = f'../data/{n_test}/op.plk'
				df = pd.DataFrame(matrix_.operation_history)
				df.to_pickle(file_path)
		
def runSQPExample(plant, cost, constraints, N, dt, solver_methods, options = {}, n_test=0, record=False):
	
	

	if constraints is None:
		trajoptMPCReference = TrajoptMPCReference(plant, cost)
	else:
		trajoptMPCReference = TrajoptMPCReference(plant, cost, constraints)

	runSolversSQP(trajoptMPCReference, N, dt, solver_methods, options, n_test, record)
 
	
	