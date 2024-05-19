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
options['path_to_urdf'] = '/home/marguerite/trajopt/TrajoptMPCReference/examples/arm.urdf'
options['RETURN_TRACE_SQP']=True
plant = URDFPlant(options=options)


N = 10
dt = 0.1



Q = np.diag([1.0,1.0,1.0,1.0])
QF = np.diag([100.0,100.0,100.0,100.0])
R = np.diag([0.1, 0.1]) 


x= np.array([0.1,0.1,0.,0.])
u= np.array([0.,0.])
xg= np.array([0.1,0.1,0.,0.])

max_radius=2
n=10
xs = np.linspace(-max_radius, max_radius, n)
ys = np.linspace(-max_radius, max_radius, n)
xgs=[[x,y, 0, 0] for x in xs for y in ys] # Cartesian coordinates 

value1=[]
value2=[]
value3=[]

grad1=[]
grad2=[]
grad3=[]


for xg in xgs:

    cost1=UrdfCost(plant,Q,QF,R,xg)
    cost2=ArmCost(Q,QF,R,xg, simplified_hessian=False)
    cost3=NumericalCost(Q,QF,R,xg)

    value1.append(cost1.value(x,u))
    value2.append(cost2.value(x,u))
    value3.append(cost3.value(x,u))


    grad1.append(cost1.gradient(x,u))
    grad2.append(cost2.gradient(x,u))
    grad3.append(cost3.gradient(x,u))


xg_range = np.arange(1, 101)

# Plotting
fig, axs = plt.subplots(3, 8, figsize=(20, 12))

# Plotting Method 1

axs[0, 0].plot(xg_range, value1, label='Cost Value')
axs[0, 0].set_title('Cost Value - Method 1')
axs[0, 0].legend()

for i in range(6):
    axs[0, i+1].plot(xg_range, [grad1[j][i] for j in range(len(xg_range))], label=f'Gradient {i+1}')
    axs[0, i+1].set_title(f'Gradient - Method 1 (Component {i+1})')
    axs[0, i+1].legend()


# Plotting Method 2
    
axs[1, 0].plot(xg_range,np.array(value2).reshape(100), label='Cost Value')
axs[1, 0].set_title('Cost Value - Method 2')
axs[1, 0].legend()

for i in range(6):
    axs[1, i+1].plot(xg_range, [grad2[j][0, i] for j in range(len(xg_range))], label=f'Gradient {i+1}')
    axs[1, i+1].set_title(f'Gradient - Method 2 (Component {i+1})')
    axs[1, i+1].legend()



# Plotting Method 3

axs[2, 0].plot(xg_range, value3, label='Cost Value')
axs[2, 0].set_title('Cost Value - Method 3')
axs[2, 0].legend()

for i in range(6):
    axs[2, i+1].plot(xg_range, [grad3[j][i] for j in range(len(xg_range))], label=f'Gradient {i+1}')
    axs[2, i+1].set_title(f'Gradient - Method 3 (Component {i+1})')
    axs[2, i+1].legend()


plt.tight_layout()
plt.savefig('cost_comparison')
plt.show()


print("URDF cost value :\n", value1)
print("Symbolic cost value :\n", value2)
print("Numerical cost value :\n", value3)


print("URDF cost gradient :\n", grad1)
print("Symbolic cost gradient :\n", grad2)
print("Numerical cost gradient :\n", grad3)



