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
     
overloading=False

n=5
type_cost='urdf'
set_hard_constraints=False
set_soft_constraints=False


N = 10
dt = 0.1

sqp_solver_methods = [SQPSolverMethods.PCG_SS] #["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]


options= {}
options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/models/arm2.urdf'
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
    # xg= np.array([-1.18, -1.58, 0.,0.]) #2
    xg= np.array([-1., 1.5, 0.,0.]) #1
    # xg= np.array([1., 1.3, 0.,0.]) #1



if(type_cost =='sym'):
    cost=ArmCost(Q,QF,R,xg,simplified_hessian=True)
elif(type_cost == 'quadratic') :
    cost=QuadraticCost(Q,QF,R,xg)
elif(type_cost == 'urdf') :
    cost=UrdfCost(plant,Q,QF,R,xg, overloading=overloading)
else:
    raise ValueError("Cost argument must be quadratic, urdf or sym")

options= {"overloading":overloading}
hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET", options=options)
soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
soft_constraints.set_torque_limits([7.0],[-7.0],"AUGMENTED_LAGRANGIAN",options=options)

if(set_hard_constraints==True):
    constraints=hard_constraints
elif(set_soft_constraints==True):
    constraints=soft_constraints
else:
    constraints=None

options = {"expected_reduction_min_SQP_DDP":-100, "RETURN_TRACE_SQP": True, "overloading": overloading} 

# options = {"expected_reduction_min_SQP_DDP":-100, "RETURN_TRACE_SQP": True, "overloading": overloading, 'DEBUG_MODE_SQP_DDP': True, 'DEBUG_MODE_Soft_Constraints':True, 'DEBUG_MODE_linSys':True} 

t1 = time.time()
# #cProfile.run('runSQPExample(plant, cost, constraints, N, dt, sqp_solver_methods, options)', sort='cumtime')
runSQPExample(plant, cost, constraints, N, dt, sqp_solver_methods, options, n_test=4, record=True)
t2 = time.time()



print(f"It took {t2-t1:.2f} seconds to compute")

# if overloading:
#     file_path = f'../analysis/data/{type_cost}/op{n_test}.plk'
#     df = pd.DataFrame(matrix_.operation_history)
#     # print(df)
#     df.to_pickle(file_path)
          
