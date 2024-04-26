#!/usr/bin/python3
from exampleHelpers import *
import cProfile
import time
import numpy as np
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt

sqp_solver_methods = [SQPSolverMethods.PCG_SS]#["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options= {}
options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/examples/2_link.urdf'
plant = URDFPlant(options=options)


N = 20
dt = 0.1


Q = np.diag([1.0,1.0,1.0,1.0])
QF = np.diag([100.0,100.0,100.0,100.0])
R = np.diag([0.1, 0.1])
xg=np.array([0.,-2.,0.,0.]) 



hard_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
hard_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")


soft_constraints = TrajoptConstraint(plant.get_num_pos(),plant.get_num_vel(),plant.get_num_cntrl(),N)
soft_constraints.set_torque_limits([7.0],[-7.0],"ACTIVE_SET")

options = {"expected_reduction_min_SQP_DDP":-100, "display": False} # needed for hard_constraints - TODO debug why

def run_test_xg(xg_):
    cost=UrdfCost(plant,Q,QF,R,xg_)
    real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)
    return real_times_sqp, cpu_times_sqp, errors_sqp

def run_test_N(N_):
    cost=UrdfCost(plant,Q,QF,R,xg)
    real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N_, dt, sqp_solver_methods, options)
    return real_times_sqp, cpu_times_sqp, errors_sqp

# def run_test_dt(dt_):
#     cost=ArmCost(Q,QF,R,xg)
#     real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt_, sqp_solver_methods, options)
#     return real_times_sqp, cpu_times_sqp, errors_sqp


def parallel_tests( xgs: np.array, Ns:np.array):
    with multiprocessing.Pool() as pool:
        results_xg = pool.map(run_test_xg, xgs)
        results_N = pool.map(run_test_N, Ns)
    return np.array(results_N),np.array(results_xg)


#Sweep feasible_set
max_radius=2
n=20
xs = np.linspace(-max_radius, max_radius, n)
ys = np.linspace(-max_radius, max_radius, n)
xgs=[[x,y, 0, 0] for x in xs for y in ys]

#Sweep N
Ns=[5,10,15,20,30,40,50] 


results_xg, results_N = parallel_tests(xgs, Ns)


x_grid, y_grid = np.meshgrid(xs, ys)
radius_grid = np.sqrt(x_grid**2 + y_grid**2)
theta_grid = np.arctan2(y_grid, x_grid)
circle_mask = radius_grid <= max_radius

real_times__xg_sqp=results_xg[:,0].reshape(n,n) * circle_mask
errors_xg_sqp = results_xg[:,2].reshape(n,n) * circle_mask


real_times_N_sqp=results_N[:,0]
errors_N_sqp=results_N[:,2]
real_times_N_mpc=results_N[:,3]
errors_N_mpc=results_N[:,5]




file_path = "results.npz"

np.savez(file_path,
         real_times__xg_sqp=real_times__xg_sqp,
         errors_xg_sqp=errors_xg_sqp,
         real_times_N_sqp=real_times_N_sqp,
         errors_N_sqp=errors_N_sqp)


# Error- Vary Xg
plt.figure(figsize=(10, 8))
sns.heatmap(errors_xg_sqp, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Position error within the feasible set for the URDF Cost, SQP, Unconstrained')
plt.savefig('plots_sqp/err_xg.png')

# CompTime- Vary Xg
plt.figure(figsize=(10, 8))
sns.heatmap(real_times__xg_sqp, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Execution time for the URDF Cost, SQP, Unconstrained')
plt.savefig('plots_sqp/rt_xg.png')


# Error- Vary N
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=errors_N_sqp, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Position error for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('plots_sqp/err_N.png')

# CompTime- Vary N
sns.set(style="whitegrid")  
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=real_times_N_sqp, marker='o')
plt.xlabel('N ')
plt.ylabel('Real Time')
plt.title('Real Time as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('plots_sqp/rt_N.png')


