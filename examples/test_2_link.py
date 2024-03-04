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


N = 5
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
    cost=QuadraticCost(Q,QF,R,xg_)
    
    real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt, sqp_solver_methods, options)
    real_times_mpc, cpu_times_mpc, errors_mpc = runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt, mpc_solver_methods, options)
    return real_times_sqp, cpu_times_sqp, errors_sqp, real_times_mpc, cpu_times_mpc, errors_mpc

def run_test_N(N_):
    cost=QuadraticCost(Q,QF,R,xg)
    real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N_, dt, sqp_solver_methods, options)
    real_times_mpc, cpu_times_mpc, errors_mpc = runMPCExample(plant, cost, hard_constraints, soft_constraints, N_, dt, mpc_solver_methods, options)
    return real_times_sqp, cpu_times_sqp, errors_sqp, real_times_mpc, cpu_times_mpc, errors_mpc

def run_test_dt(dt_):
    cost=QuadraticCost(Q,QF,R,xg)
    real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt_, sqp_solver_methods, options)
    real_times_mpc, cpu_times_mpc, errors_mpc = runMPCExample(plant, cost, hard_constraints, soft_constraints, N, dt_, mpc_solver_methods, options)
    return real_times_sqp, cpu_times_sqp, errors_sqp, real_times_mpc, cpu_times_mpc, errors_mpc


def parallel_tests(xgs:np.array, Ns:np.array, dts: np.array):

    with multiprocessing.Pool() as pool:
        #results_xg = pool.map(run_test_xg, xgs)
        results_N = pool.map(run_test_xg, Ns)
        results_dt = pool.map(run_test_xg, dts)
    return np.array(results_N), np.array(results_dt)


#Sweep feasible_set
max_radius=2
n=20
xs = np.linspace(-max_radius, max_radius, n)
ys = np.linspace(-max_radius, max_radius, n)
xgs=[[x,y, 0, 0] for x in xs for y in ys]
#Sweep N
Ns=[i for i in range(1, 31)] #30
#Sweep dt
dts=[0.01, 0.1, 0.2, 0.5, 1., 1.5] #6
results_xg, results_N, results_dt = parallel_tests(xgs, Ns, dts)





x_grid, y_grid = np.meshgrid(xs, ys)
radius_grid = np.sqrt(x_grid**2 + y_grid**2)
theta_grid = np.arctan2(y_grid, x_grid)
circle_mask = radius_grid <= max_radius

# real_times__xg_sqp=results_xg[:,0].reshape(n,n) * circle_mask
# cpu_times_xg_sqp= results_xg[:,1].reshape(n,n) * circle_mask
# errors_xg_sqp = results_xg[:,2].reshape(n,n) * circle_mask
# real_times__xg_mpc=results_xg[:,3].reshape(n,n) * circle_mask
# cpu_times_xg_mpc= results_xg[:,4].reshape(n,n) * circle_mask
# errors_xg_mpc = results_xg[:,5].reshape(n,n) * circle_mask


real_times_N_sqp=results_N[:,0]
cpu_times_N_sqp=results_N[:,1]
errors_N_sqp=results_N[:,2]
real_times_N_mpc=results_N[:,3]
cpu_times_N_mpc=results_N[:,4]
errors_N_mpc=results_N[:,5]

real_times_dt_sqp=results_dt[:,0]
cpu_times_dt_sqp=results_dt[:,1]
errors_dt_sqp=results_dt[:,2]
real_times_dt_mpc=results_dt[:,3]
cpu_times_dt_mpc=results_dt[:,4]
errors_dt_mpc=results_dt[:,5]

file_path = "results.npz"

# Save all arrays to the same compressed file
np.savez(file_path,
        #  real_times__xg_sqp=real_times__xg_sqp,
        #  cpu_times_xg_sqp=cpu_times_xg_sqp,
        #  errors_xg_sqp=errors_xg_sqp,
        #  real_times__xg_mpc=real_times__xg_mpc,
        #  cpu_times_xg_mpc=cpu_times_xg_mpc,
        #  errors_xg_mpc=errors_xg_mpc,
         real_times_N_sqp=real_times_N_sqp,
         cpu_times_N_sqp=cpu_times_N_sqp,
         errors_N_sqp=errors_N_sqp,
         real_times_N_mpc=real_times_N_mpc,
         cpu_times_N_mpc=cpu_times_N_mpc,
         errors_N_mpc=errors_N_mpc,
         real_times_dt_sqp=real_times_dt_sqp,
         cpu_times_dt_sqp=cpu_times_dt_sqp,
         errors_dt_sqp=errors_dt_sqp,
         real_times_dt_mpc=real_times_dt_mpc,
         cpu_times_dt_mpc=cpu_times_dt_mpc,
         errors_dt_mpc=errors_dt_mpc)




# plt.figure(figsize=(10, 8))
# sns.heatmap(errors_xg_sqp, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'Position error within the feasible set for the Arm Cost, SQP, Unconstrained, resolution of {n}')
# plt.savefig('plots/err_xg_sqp_.png')

# plt.figure(figsize=(10, 8))
# sns.heatmap(errors_xg_mpc, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'Position error within the feasible set for the Arm Cost, MPC, Unconstrained, resolution of {n}')
# plt.savefig('plots/err_xg_mpc_.png')

# plt.figure(figsize=(10, 8))
# sns.heatmap(real_times__xg_sqp, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_a    mask)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'Execution time for the Arm Cost, SQP, Unconstrained,resolution of {n}')
# plt.savefig('plots/rt_xg_sqp_.png')

# plt.figure(figsize=(10, 8))
# sns.heatmap(real_times__xg_mpc, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'Execution time for the Arm Cost, MPC, Unconstrained, resolution of {n}')
# plt.savefig('plots/rt_xg_mpc_.png')

# plt.figure(figsize=(10, 8))
# sns.heatmap(cpu_times_xg_sqp, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'CPU time for the Arm Cost, SQP, Unconstrained,resolution of {n}')
# plt.savefig('plots/cput_xg_sqp_.png')

# plt.figure(figsize=(10, 8))
# sns.heatmap(cpu_times_xg_mpc, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(f'CPU time for the Arm Cost, MPC, Unconstrained, resolution of {n}')
# plt.savefig('plots/cput_xg_mpc_.png')


sns.set(style="whitegrid")  
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=real_times_N_sqp, marker='o')
plt.xlabel('N ')
plt.ylabel('Real Time')
plt.title('Real Time as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('plots/rt_N_sqp_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=real_times_N_mpc, marker='o')
plt.xlabel('N ')
plt.ylabel('Real Time')
plt.title('Real Time as a Function of N - MPC - Unconstrained')
plt.legend()
plt.savefig('plots/rt_N_mpc_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=cpu_times_N_sqp, marker='o')
plt.xlabel('N ')
plt.ylabel('CPU Time')
plt.title('CPU Time as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('plots/cput_N_sqp_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=cpu_times_N_mpc, marker='o')
plt.xlabel('N ')
plt.ylabel('CPU Time')
plt.title('CPU Time as a Function of N - MPC - Unconstrained')
plt.legend()
plt.savefig('plots/cput_N_mpc_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=errors_N_sqp, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Position error for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('plots/erros_N_sqp_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=errors_N_mpc, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Position error for the Arm Cost as a Function of N - MPC - Unconstrained')
plt.legend()
plt.savefig('plots/erros_N_mpc_.png')


plt.figure(figsize=(10, 6))  
sns.lineplot(x=dts, y=real_times_dt_sqp, marker='o')
plt.xlabel('dt ')
plt.ylabel('Real Time')
plt.title('Real Time as a Function of the timestep - SQP - Unconstrained')
plt.legend()
plt.savefig('plots/rt_dt_sqp_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=dts, y=real_times_dt_mpc, marker='o')
plt.xlabel('dt ')
plt.ylabel('Real Time')
plt.title('Real Time as a Function of the timestep - MPC - Unconstrained')
plt.legend()
plt.savefig('plots/rt_dt_mpc_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=dts, y=cpu_times_dt_sqp, marker='o')
plt.xlabel('dt ')
plt.ylabel('CPU Time')
plt.title('CPU Time as a Function of the timestep - SQP - Unconstrained')
plt.legend()
plt.savefig('plots/cput_dt_sqp_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=dts, y=cpu_times_dt_mpc, marker='o')
plt.xlabel('dt ')
plt.ylabel('CPU Time')
plt.title('CPU Time as a Function of the timestep - MPC - Unconstrained')
plt.legend()
plt.savefig('plots/cput_dt_mpc_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=dts, y=errors_dt_sqp, marker='o')
plt.xlabel('dt ')
plt.ylabel('Error')
plt.title('Position error for the Arm Cost as a Function of the timestep - SQP - Unconstrained')
plt.legend()
plt.savefig('plots/erros_dt_sqp_.png')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=dts, y=errors_dt_mpc, marker='o')
plt.xlabel('dt ')
plt.ylabel('Error ')
plt.title('Position error for the Arm Cost as a Function of the timestep - MPC - Unconstrained')
plt.legend()
plt.savefig('plots/erros_dt_mpc_.png')