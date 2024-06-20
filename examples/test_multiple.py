#!/usr/bin/python3
from exampleHelpers import *
import cProfile
import time
import numpy as np
import multiprocessing
import seaborn as sns
import sympy

import csv
import sys
import pandas as pd

from overloading import matrix_

overloading = True

N = 10
dt = 0.1
n=20

id_start=4
id_stop=20

max_radius=2
xs = np.linspace(-max_radius, max_radius, n)
ys = np.linspace(-max_radius, max_radius, n)
xgs=[[x,y, 0, 0] for x in xs for y in ys]
xg_sweep = [[x, y, 0, 0] for x in xs for y in ys if x**2 + y**2 <= max_radius**2] # Filter out points that are inside the square but not the cirle 

settings_df = pd.read_csv('test_settings.csv')
sweep=(settings_df.loc[id_start]['xg']=='Sweep')
sweep=True
if sweep:
    id_stop=len(xg_sweep)+id_start

default_xg= [1, 1.5, 0., 0.]

def run_test(test_settings):
    n_test = test_settings['Test number']+3
    n_links = 2#test_settings['number of links']
    xg = test_settings['xg']
    type_cost = 'URDF'#test_settings['type of Cost']
    N = 10#test_settings['N']
    set_hard_constraints = False#test_settings['Set Hard constraints']
    set_soft_constraints = False# test_settings['Soft constraints']
    hessian = 'approximate'# test_settings['Hessian']
   
    
    if sweep:
        xg=xg_sweep[n_test]
    else:
        xg=default_xg

    sqp_solver_methods = [SQPSolverMethods.PCG_SS]
    options = {}
    options['path_to_urdf'] = f'/home/marguerite/Documents/lab/TrajoptMPCReference/models/arm{n_links}.urdf'
    options['overloading'] = overloading
    plant = URDFPlant(options=options)

    if overloading:
        Q = matrix_([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])
        QF = matrix_([[100.0, 0.0, 0.0, 0.0],
                      [0.0, 100.0, 0.0, 0.0],
                      [0.0, 0.0, 100.0, 0.0],
                      [0.0, 0.0, 0.0, 100.0]])
        R = matrix_([[0.1, 0.0],
                     [0.0, 0.1]])
        xg = matrix_([-1.18, -1.58, 0.,0.])
    else:
        Q = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        QF = np.array([[100.0, 0.0, 0.0, 0.0],
                       [0.0, 100.0, 0.0, 0.0],
                       [0.0, 0.0, 100.0, 0.0],
                       [0.0, 0.0, 0.0, 100.0]])
        R = np.array([[0.1, 0.0],
                      [0.0, 0.1]])
        xg = np.array([1, 1.5, 0., 0.])

    if type_cost == 'Symbolic':
        cost = ArmCost(Q, QF, R, xg, simplified_hessian=True)
    elif type_cost == 'Quadratic':
        cost = QuadraticCost(Q, QF, R, xg)
    elif type_cost == 'URDF':
        cost = UrdfCost(plant, Q, QF, R, xg, overloading=overloading)
        if(hessian=='approximate'):
            cost.hess_mode=0
        elif(hessian=='exacte'):
            cost.hess_mode=1
        elif(hessian=='grad.T*grad'):
            cost.hess_mode=2
        elif(hessian=='No Hessian'):
            cost.hess_mode=3
        else:
            cost.hess_mode=0
    else:
        raise ValueError("Cost argument must be quadratic, urdf or sym")

    
    options = {"overloading": overloading}
    hard_constraints = TrajoptConstraint(plant.get_num_pos(), plant.get_num_vel(), plant.get_num_cntrl(), N)
    hard_constraints.set_torque_limits([7.0], [-7.0], "ACTIVE_SET", options=options)
    soft_constraints = TrajoptConstraint(plant.get_num_pos(), plant.get_num_vel(), plant.get_num_cntrl(), N)
    soft_constraints.set_torque_limits([7.0], [-7.0], "AUGMENTED_LAGRANGIAN", options=options)

    if set_hard_constraints:
        constraints = hard_constraints
    elif set_soft_constraints:
        constraints = soft_constraints
    else:
        constraints = None

    options = {"expected_reduction_min_SQP_DDP": -100, "overloading": overloading}

    runSQPExample(plant, cost, constraints, N, dt, sqp_solver_methods, options, n_test, record=True)

def parallel_tests(settings_df):
    n_tot_test= id_stop-id_start
    test_settings_list = [settings_df.iloc[i].to_dict() for i in range(id_start,id_stop)]
    
    with multiprocessing.Pool(processes=n_tot_test) as pool:
        pool.map(run_test, test_settings_list)

if __name__ == "__main__":
    parallel_tests(settings_df)
