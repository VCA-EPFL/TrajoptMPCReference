#!/usr/bin/python3
from exampleHelpers import *
import cProfile
import time
import numpy as np
import multiprocessing
import seaborn as sns
import sympy
import matplotlib.pyplot as plt




sqp_solver_methods = [SQPSolverMethods.PCG_SS]#["N", "S", "PCG-J", "PCG-BJ", "PCG-SS"]
mpc_solver_methods = [MPCSolverMethods.QP_PCG_SS] #["iLQR", "QP-N", "QP-S", "QP-PCG-J", "QP-PCG-BJ", "QP-PCG-SS"]

options= {}
# options['path_to_urdf'] = '/home/marguerite/Documents/lab/TrajoptMPCReference/examples/arm.urdf'
options['path_to_urdf'] = '/home/marguerite/trajopt/TrajoptMPCReference/examples/arm3.urdf'
options['RETURN_TRACE_SQP']=True
plant = URDFPlant(options=options)


N = 5
dt = 0.1

    


Q = np.diag([1.0,1.0,1.0,1.0]) #stays (4,4)
QF = np.diag([100.0,100.0,100.0,100.0])
R = np.diag([0.1, 0.1, 0.1]) #size = n_joints


x= np.array([0.1,0.1,0.1,0.,0., 0.]) #size = 2*n_joints
u= np.array([1.,3., 1.]) #size = n_joints
xg= np.array([-2,0.2, 4.,5.]) #Stays size (4,)

cost1=UrdfCost(plant,Q,QF,R,xg)
t1 = time.time()
print("Cost urdf:", cost1.value(x,u))
print("Grad urdf:", cost1.gradient(x,u), type(cost1.gradient(x,u)), cost1.gradient(x,u).shape)
print("Hess urdf: \n", cost1.hessian(x,u), type(cost1.hessian(x,u)))
t2 = time.time()
print(f"It took {t2-t1:.2f} seconds to compute")

# cost2=ArmCost(Q,QF,R,xg, simplified_hessian=False)

# t3 = time.time()
# print("Cost sym:", cost2.value(x,u))
# print("Grad sym:", cost2.gradient(x,u), type(cost2.gradient(x,u)), cost2.gradient(x,u).shape)
# print("Hess sym:\n", cost2.hessian(x,u))
# t4 = time.time()

# print(f"It took {t4-t3:.2f} seconds to compute")





# cost3=NumericalCost(Q,QF,R,xg)
# print("Cost num:", cost3.value(x,u))
# print("Grad num:", cost3.gradient(x,u))
# print("Hess num:\n", cost3.hessian(x,u))

# max_radius=2
# n=5
# xs = np.linspace(-max_radius, max_radius, n)
# ys = np.linspace(-max_radius, max_radius, n)
# xgs=[[x,y, 0, 0] for x in xs for y in ys] # Cartesian coordinates 

# value1=[]
# value2=[]
# value3=[]

# grad1=[]
# grad2=[]
# grad3=[]


# for xg in xgs:

#     cost1=UrdfCost(plant,Q,QF,R,xg)
#     cost2=ArmCost(Q,QF,R,xg, simplified_hessian=False)
#     cost3=NumericalCost(Q,QF,R,xg)

#     value1.append(cost1.value(x,u))
#     value2.append(cost2.value(x,u))
#     value3.append(cost3.value(x,u))


#     grad1.append(cost1.gradient(x,u))
#     grad2.append(cost2.gradient(x,u))
#     grad3.append(cost3.gradient(x,u))


# xg_range = np.arange(1, n*n + 1)

# # Plotting
# fig, axs = plt.subplots(3, 8, figsize=(20, 12))

# # Plotting Method 1

# axs[0, 0].plot(xg_range, value1, label='Cost Value')
# axs[0, 0].set_title('Cost Value - URDF')
# axs[0, 0].legend()

# for i in range(6):
#     axs[0, i+1].plot(xg_range, [grad1[j][i] for j in range(len(xg_range))], label=f'Element {i+1}')
#     axs[0, i+1].set_title(f'Gradient - URDF (Component {i+1})')
#     axs[0, i+1].legend()


# # Plotting Method 2
    
# axs[1, 0].plot(xg_range,value2, label='Cost Value')
# axs[1, 0].set_title('Cost Value - Symbolic')
# axs[1, 0].legend()

# for i in range(6):
#     axs[1, i+1].plot(xg_range, [grad2[j][i] for j in range(len(xg_range))], label=f'Element {i+1}')
#     axs[1, i+1].set_title(f'Gradient - Symbolic (Component {i+1})')
#     axs[1, i+1].legend()



# # Plotting Method 3

# axs[2, 0].plot(xg_range, value3, label='Cost Value')
# axs[2, 0].set_title('Cost Value - Numerical')
# axs[2, 0].legend()

# for i in range(6):
#     axs[2, i+1].plot(xg_range, [grad3[j][i] for j in range(len(xg_range))], label=f'Element {i+1}')
#     axs[2, i+1].set_title(f'Gradient - Numerical (Component {i+1})')
#     axs[2, i+1].legend()


# plt.tight_layout()
# plt.savefig('results/cost_comparison')
# plt.show()


# # print("URDF cost value :\n", value1)
# # print("Symbolic cost value :\n", value2)
# # print("Numerical cost value :\n", value3)


# # print("URDF cost gradient :\n", grad1)
# # print("Symbolic cost gradient :\n", grad2)
# # print("Numerical cost gradient :\n", grad3)



