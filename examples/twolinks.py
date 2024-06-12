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
import sys
     
overloading=True
type_cost='urdf'
set_hard_constraints=False
set_soft_constraints=False
# type_cost=sys.argv[1]
N = 10
dt = 0.1

sqp_solver_methods = [SQPSolverMethods.PCG_SS] #["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options= {}
options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/examples/arm.urdf'
# options['path_to_urdf'] = '/home/marguerite/trajopt/TrajoptMPCReference/examples/arm.urdf'
options['DEBUG_MODE_SQP_DDP']= True
options['overloading']=overloading

plant = URDFPlant(options=options)


if(overloading):
    Q=matrix_([[1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])
    QF=matrix_([[100.0, 0.0, 0.0, 0.0],
    [0.0, 100.0, 0.0, 0.0],
    [0.0, 0.0, 100.0, 0.0],
    [0.0, 0.0, 0.0, 100.0]])
    R=matrix_([[0.1, 0.0],
    [0.0, 0.1]])
    xg= matrix_([-1., 1.5, 0.,0.])
else:
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
    xg= np.array([-1., 1.5, 0.,0.])


if(type_cost =='sym'):
    cost=ArmCost(Q,QF,R,xg,simplified_hessian=True)
elif(type_cost == 'quadratic') :
    cost=QuadraticCost(Q,QF,R,xg)
elif(type_cost == 'urdf') :
    cost=UrdfCost(plant,Q,QF,R,xg, overloading=overloading)
else:
    raise ValueError("Cost argument must be quadratic, urdf or sym")

# cost1=UrdfCost(plant,Q,QF,R,xg, overloading=False)

# x= np.array([0.1,0.1,0.1,0.1]) #size = 2*n_joints
# u= np.array([1.,3.]) #size = n_joints
# xg= np.array([-2,0.2, 4.,5.]) #Stays size (4,)
# print("Cost urdf: \n", cost1.value(x,u), type(cost1.value(x,u)))
# print("Grad urdf: \n", cost1.gradient(x,u), type(cost1.gradient(x,u)), cost1.gradient(x,u).shape)
# print("Hess urdf: \n", cost1.hessian(x,u))


# x2= matrix_([0.1,0.1,0.1,0.1]) #size = 2*n_joints
# u2= matrix_([1.,3.]) #size = n_joints
# xg2= matrix_([-2,0.2, 4.,5.]) #Stays size (4,)
# print("Cost urdf 2: \n", cost2.value(x2,u2),type(cost2.value(x,u)))
# print("Grad urdf 2: \n", cost2.gradient(x2,u2), type(cost2.gradient(x2,u2)), cost2.gradient(x2,u2).shape)
# print("Hess urdf 2: \n", cost2.hessian(x2,u2))

if(set_hard_constraints):
    hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
    hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")
else:
    hard_constraints=None

if(set_soft_constraints):
    soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
    soft_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")
else:
    soft_constraints=None

options = {"expected_reduction_min_SQP_DDP":-100, "display": True, "overloading": overloading} 

t1 = time.time()
# #cProfile.run('runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)', sort='cumtime')
error= runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)
t2 = time.time()

print(f"It took {t2-t1:.2f} seconds to compute")

# current_frame = inspect.currentframe()
# innerframes=inspect.getinnerframes(current_frame)
# print("Outer frames: ", [(frame.filename, frame.function, frame.lineno) for frame,_,_,_,_ in innerframes])

# runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options)


# csv_file_path = f'data/{type_cost}/cost.csv'
# with open(csv_file_path, 'w', newline='\n') as file:
#     csv_writer = csv.writer(file)
#     csv_writer.writerows(cost.saved_cost)

# csv_file_path = f'data/{type_cost}/gradient.csv'
# with open(csv_file_path, 'w', newline='\n') as file:
#     csv_writer = csv.writer(file)
#     csv_writer.writerows(cost.saved_grad)

if overloading:
    csv_file_path = f'../analysis/data/{type_cost}/op1.csv'
    with open(csv_file_path, 'w', newline='\n') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix_.operation_history)