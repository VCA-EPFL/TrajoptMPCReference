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

overloading = False

N = 10
dt = 0.1

settings_df = pd.read_csv('test_settings.csv')
def run_test(test_settings):
    n_test = 1#test_settings['Test number']
    n_links = test_settings['number of links']
    xg = test_settings['xg']
    type_cost = test_settings['type of Cost']
    N = test_settings['N']
    set_hard_constraints = False#test_settings['Set Hard constraints']
    set_soft_constraints = False# test_settings['Soft constraints']
    hessian = 0# test_settings['Hessian']
    integrator_type = test_settings['Integrator type']
    lin_sys_method = test_settings['Lin Sys method']
    use_pcg = test_settings['Use PCG']
    minv = test_settings['Minv']

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
        xg = matrix_([-1., 1.5, 0., 0.])
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
    n_tot_test=1#len(settings_df)
    test_settings_list = [settings_df.iloc[i].to_dict() for i in range(n_tot_test)]

    with multiprocessing.Pool(processes=n_tot_test) as pool:
        pool.map(run_test, test_settings_list)

if __name__ == "__main__":
    parallel_tests(settings_df)
