#!/usr/bin/python3
from exampleHelpers import *
import cProfile
import time
import numpy as np
import multiprocessing
import seaborn as sns
import sympy

import csv
from overloading import matrix_

     

sqp_solver_methods = [SQPSolverMethods.PCG_SS] #["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options= {}
# options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/examples/arm.urdf'
options['path_to_urdf'] = '/home/marguerite/trajopt/TrajoptMPCReference/examples/arm.urdf'

# options['path_to_urdf'] = '/../Documents/lab/TrajoptMPCReference/examples/2_link.urdf'

plant = URDFPlant(options=options)


N = 20
dt = 0.1



Q=np.array([[1.0, 0.0, 0.0, 0.0],
 [0.0, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.0],
 [0.0, 0.0, 0.0, 1.0]])
QF=np.array([[100.0, 0.0, 0.0, 0.0],
 [0.0, 100.0, 0.0, 0.0],
 [0.0, 0.0, 100.0, 0.0],
 [0.0, 0.0, 0.0, 100.0]])
R=np.array([[0.1, 0.0],
 [0.0, 0.1]])
xg= np.array([2., 1.0, 0.,0.])



cost=ArmCost(Q,QF,R,xg,simplified_hessian=True)
# cost=UrdfCost(plant,Q,QF,R,xg)







hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")


soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
soft_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

options = {"expected_reduction_min_SQP_DDP":-100, "display": False} # needed for hard_constraints - TODO debug why

t1 = time.time()
# #cProfile.run('runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)', sort='cumtime')
runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)
t2 = time.time()

print(f"It took {t2-t1:.2f} seconds to compute")

# current_frame = inspect.currentframe()
# innerframes=inspect.getinnerframes(current_frame)
# print("Outer frames: ", [(frame.filename, frame.function, frame.lineno) for frame,_,_,_,_ in innerframes])

# runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options)

csv_file_path = 'cost.csv'
with open(csv_file_path, 'w', newline='\n') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(cost.saved_cost)

csv_file_path = 'gradient.csv'
with open(csv_file_path, 'w', newline='\n') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(cost.saved_grad)

csv_file_path = 'hessian.csv'
with open(csv_file_path, 'w', newline='\n') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(cost.saved_hess)