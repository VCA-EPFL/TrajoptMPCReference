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
options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/examples/arm.urdf'
plant = URDFPlant(options=options)


N = 6
dt = 0.1


Q = np.diag([1.0,1.0,1.0,1.0])
QF = np.diag([100.0,100.0,100.0,100.0])
R = np.diag([0.1, 0.1])
xg=np.array([0.1,0.1,0.,0.]) 



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
    print("N is: ", N_)
    cost=UrdfCost(plant,Q,QF,R,xg)
    real_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N_, dt, sqp_solver_methods, options)
    return real_times_sqp, errors_sqp

# def run_test_dt(dt_):
#     cost=ArmCost(Q,QF,R,xg)
#     real_times_sqp, cpu_times_sqp, errors_sqp = runSQPExample(plant, cost, hard_constraints, soft_constraints, N, dt_, sqp_solver_methods, options)
#     return real_times_sqp, cpu_times_sqp, errors_sqp


def parallel_tests( xgs: np.array, Ns:np.array):
    with multiprocessing.Pool() as pool:
        results_xg = pool.map(run_test_xg, xgs)
        print("Finished Test Xg")
        results_N = pool.map(run_test_N, Ns)
        print("Finished Test Ns")
    return results_N, results_xg



# Sweep feasible_set 
max_radius=2
n=20
xs = np.linspace(-max_radius, max_radius, n)
ys = np.linspace(-max_radius, max_radius, n)
xgs=[[x,y, 0, 0] for x in xs for y in ys] # Cartesian coordinates 

# Mask for xg plot
x_grid, y_grid = np.meshgrid(xs, ys)
radius_grid = np.sqrt(x_grid**2 + y_grid**2)
theta_grid = np.arctan2(y_grid, x_grid)
circle_mask = radius_grid <= max_radius

#Sweep N
Ns=[5,10,15,20,30,40,50] 

results_N, results_xg = parallel_tests(xgs, Ns)
#results_N = [(24.952503303999947, np.array([-0.0875,0.9 , -0.0152, 0.])), (154.61594967700012, np.array([-0.0169, 0.9 , -0.0013,  0. ]))]
#results_xg= [(47.88541344100008, np.array([1.9859, 3. , 0.2639, 0.])), (47.680899413000134, np.array([ 1.9859,-1., 0.2639, 0.])), (47.70890542400002, np.array([-1.9859, 3., -0.2639,  0.])), (47.67314716300007, np.array([-1.9859, -1., -0.2639,  0. ]))]
print("Plotting")
time_N=np.array([result[0] for result in results_N])
err_N=np.array([result[1] for result in results_N])
err_x_N=err_N[:,0]
err_y_N=err_N[:,1]
err_vx_N=err_N[:,2]
err_vy_N=err_N[:,3]
err_norm_N=np.array([np.linalg.norm(err) for err in err_N])


time_xg=np.array([result[0] for result in results_xg]).reshape(n,n) * circle_mask
err_xg=np.array([result[1] for result in results_xg])
err_x_xg=err_xg[:,0].reshape(n,n) * circle_mask
err_y_xg=err_xg[:,1].reshape(n,n) * circle_mask
err_vx_xg=err_xg[:,2].reshape(n,n)* circle_mask
err_vy_xg=err_xg[:,3].reshape(n,n)* circle_mask
err_norm_xg=np.array([np.linalg.norm(err) for err in err_xg]).reshape(n,n) * circle_mask
# print("Result: ", results_xg)
print("Error: ", err_xg)
print("Error x ", err_x_xg)
# print("Error y ", err_y_xg)
# print("Error vx ", err_vx_xg)
# print("Error vy ", err_vy_xg)
# print("Error norm ", err_norm_xg)









# real_times_xg=results_xg[:,0].reshape(n,n) * circle_mask
# errors_xg = results_xg[:,2].reshape(n,n) * circle_mask







file_path = "results_urdf.npz"

np.savez(file_path,
         time_xg=time_xg,
         err_xg=err_xg,
         time_N=time_N,
         err_N=err_N)



# Error- Vary Xg
plt.figure(figsize=(10, 8))
sns.heatmap(err_x_xg, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Error in x within the feasible set for the URDF Cost, SQP, Unconstrained')
plt.savefig('error_x_xg')
plt.figure(figsize=(10, 8))
sns.heatmap(err_y_xg, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Error in y within the feasible set for the URDF Cost, SQP, Unconstrained')
plt.savefig('error_y_xy')
plt.figure(figsize=(10, 8))
sns.heatmap(err_vx_xg, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Error in vx within the feasible set for the URDF Cost, SQP, Unconstrained')
plt.savefig('error_vx_xg')
plt.figure(figsize=(10, 8))
sns.heatmap(err_vy_xg, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Error in vy within the feasible set for the URDF Cost, SQP, Unconstrained')
plt.savefig('errorvy_xg')
plt.figure(figsize=(10, 8))
sns.heatmap(err_norm_xg, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Error within the feasible set for the URDF Cost, SQP, Unconstrained')
plt.savefig('error_norm_xg')


# CompTime- Vary Xg
plt.figure(figsize=(10, 8))
sns.heatmap(time_xg, cmap="viridis", xticklabels=False, yticklabels=False, mask=~circle_mask)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Execution time for the URDF Cost, SQP, Unconstrained')
plt.savefig('time_xg')


# Error- Vary N
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=err_x_N, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Error in x for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('err_x_xg')
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=err_y_N, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Error in y for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('err_y_xg')
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=err_vx_N, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Error in vx for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('err_vx_xg')
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=err_vy_N, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Error in vy for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('err_vy_xg')

plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=err_norm_N, marker='o')
plt.xlabel('N ')
plt.ylabel('Error')
plt.title('Error in vy for the Arm Cost as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('')

# CompTime- Vary N
sns.set(style="whitegrid")  
plt.figure(figsize=(10, 6))  
sns.lineplot(x=Ns, y=time_N, marker='o')
plt.xlabel('N ')
plt.ylabel('Real Time')
plt.title('Real Time as a Function of N - SQP - Unconstrained')
plt.legend()
plt.savefig('time_N')


