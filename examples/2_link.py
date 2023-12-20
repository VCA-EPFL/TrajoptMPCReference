#!/usr/bin/python3
from exampleHelpers import *

sqp_solver_methods = [SQPSolverMethods.PCG_SS]#["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options= {}
options['path_to_urdf'] = '/home/tbourgea/git/OurClone/examples/2_link.urdf'
plant = URDFPlant(options=options)


N = 20
dt = 0.1

Q = np.diag([1.0,1.0, 1.0, 1.0])
QF = np.diag([100.0,100.0, 100., 100.])
R = np.diag([0.1, 0.1])
xg = np.array([1.2,1.23,0,0])
cost = QuadraticCost(Q,QF,R,xg)

hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
soft_constraints.set_torque_limits([7.0],[-7.0],"AUGMENTED_LAGRANGIAN")

options = {"expected_reduction_min_SQP_DDP":-100, "display": True, "gravity" : -27} # needed for hard_constraints - TODO debug why

runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)

runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options)