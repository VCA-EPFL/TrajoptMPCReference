#!/usr/bin/python3
from exampleHelpers import *
import cProfile
import time
import numpy as np
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import json

sqp_solver_methods = [SQPSolverMethods.PCG_SS]#["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options= {}
# options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/examples/arm.urdf'
options['path_to_urdf'] = '/home/marguerite/trajopt/TrajoptMPCReference/examples/arm.urdf'
plant = URDFPlant(options=options)

type_cost=sys.argv[1]
N = 6
dt = 0.1


Q = np.diag([1.0,1.0,1.0,1.0])
QF = np.diag([100.0,100.0,100.0,100.0])
R = np.diag([0.1, 0.1])
xg=np.array([-1,1.5,0.,0.]) 



hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")


soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
soft_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

options = {"expected_reduction_min_SQP_DDP":-100, "display": False} # needed for hard_constraints - TODO debug why


def run_test_xg(xg_):
    cost=UrdfCost(plant,Q,QF,R,xg_)
    real_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)
    return real_times_sqp, errors_sqp

def run_test_N(N_):
    t1 = time.perf_counter()
    cost=ArmCost(Q,QF,R,xg, simplified_hessian=True)
    t2 = time.perf_counter()
    errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N_, dt, sqp_solver_methods, options)
    t3 = time.perf_counter()
    return t2-t1, errors_sqp

# def run_test_dt(dt_):
#     cost=ArmCost(Q,QF,R,xg)
#     real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt_, sqp_solver_methods, options)
#     return real_times_sqp, cpu_times_sqp, errors_sqp


def parallel_tests(Ns:np.array):
    with multiprocessing.Pool() as pool:
        results_xg = pool.map(run_test_xg, xgs)
        #results_N = pool.map(run_test_N, Ns)
    return results_xg

def save_results(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


# Sweep feasible_set 
max_radius=2
n=20
#sweep square 
# xs = np.linspace(-max_radius, max_radius, n)
# ys = np.linspace(-max_radius, max_radius, n)
# xgs=[[x,y, 0, 0] for x in xs for y in ys]
# xgs = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2] # Filter out points that are inside the square but not the cirle 


# Local set
xs = np.linspace(-1.5, -0.5, n)
ys = np.linspace(-2, -1, n)
xgs = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2]


results_xg = parallel_tests(xgs)

time_xg=np.array([result[0] for result in results_xg])
err_xg=np.array([result[1] for result in results_xg])
time_xg_list = time_xg.tolist()
err_xg_list = err_xg.tolist()
save_results(f'results/{type_cost}/results_xg_err.json', err_xg_list)
save_results(f'results/{type_cost}/results_xg_time.json', time_xg_list)














