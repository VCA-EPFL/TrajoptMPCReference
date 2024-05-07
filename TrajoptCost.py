import numpy as np
from sympy import symbols, diff, Matrix, cos, sin, MatrixSymbol, BlockMatrix, lambdify, ccode
from expressions import *
from TrajoptPlant import *





class TrajoptCost:
	def value():
		raise NotImplementedError

	def gradient():
		raise NotImplementedError

	def hessian():
		raise NotImplementedError

class QuadraticCost(TrajoptCost):
	def __init__(self, Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None):
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start

	def get_currQ(self, u = None, timestep = None):
		last_state = isinstance(u,type(None))
		shifted_QF = (not isinstance(timestep,type(None)) \
			          and not isinstance(self.QF_start,type(None)) \
			          and timestep >= self.QF_start)
		use_QF =  last_state or shifted_QF
		currQ = self.QF if use_QF else self.Q
		return currQ

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,timestep)
		cost = 0.5*np.matmul(delta_x.transpose(),np.matmul(currQ,delta_x))
		if not isinstance(u, type(None)):
			cost += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		return cost

	def gradient(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,timestep)
		top = np.matmul(delta_x.transpose(),currQ)

		if u is None:
			return top
		else:
			bottom = np.matmul(u.transpose(),self.R)
			return np.hstack((top,bottom))

	def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		nx = self.Q.shape[0]
		nu = self.R.shape[0]
		currQ = self.get_currQ(u,timestep)

		if u is None:
			return currQ
		else:
			top = np.hstack((currQ,np.zeros((nx,nu))))
			bottom = np.hstack((np.zeros((nu,nx)),self.R))
			return np.vstack((top,bottom))

	def increase_QF(self, multiplier: float = 2.0):
		self.QF *= multiplier
		self.increaseCount_QF += 1
		return self.increaseCount_QF

	def increase_Q(self, multiplier: float = 2.0):
		self.Q *= multiplier
		self.increaseCount_Q += 1
		return self.increaseCount_Q

	def reset_increase_count_QF(self):
		self.increaseCount_QF = 0

	def reset_increase_count_Q(self):
		self.increaseCount_Q = 0

	def shift_QF_start(self, shift: float = -1.0):
		self.QF_start += shift
		self.QF_start = max(self.QF_start, 0)
		return self.QF_start

class ArmCost(TrajoptCost):

	def __init__(self, Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None):
		
		
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start
		self.l1=1
		self.l2=1
		

		self.cost_control_in=self.symbolic_cost(control=True)
		self.cost_control_in_eval=self.symbolic_cost_eval(control=True)
		self.cost_control_off=self.symbolic_cost(control=False)
		self.cost_control_off_eval=self.symbolic_cost_eval(control=False)
		self.grad_control_in=self.symbolic_gradient(control=True)
		self.grad_control_off=self.symbolic_gradient(control=False)
		self.grad_control_in_eval=self.symbolic_gradient_eval(control=True)
		self.grad_control_off_eval=self.symbolic_gradient_eval(control=False)
		# self.hess_control_in=self.symbolic_hessian(control=True)
		# self.hess_control_off=self.symbolic_hessian(control=False)
		# self.hess_control_in_eval=self.symbolic_hessian_eval(control=True)
		# self.hess_control_off_eval=self.symbolic_hessian_eval(control=False)




	def symbolic_cost(self, control=True):

		q1,q2,q1_dot,q2_dot, u1, u2  = symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)
		

		# x=self.l2*cos(q2+q1)+self.l1*cos(q1)
		# y=self.l2*sin(q2+q1)+self.l1*sin(q1)

		x=-self.l2*sin(q2+q1)-self.l1*sin(q1)
		y=self.l2*cos(q2+q1)+self.l1*cos(q1)

		
		# J = Matrix([
		# 	[-self.l2*sin(q2+q1)-self.l1*sin(q1), -self.l2*sin(q2+q1)],
		# 	[self.l2*cos(q2+q1)-self.l1*cos(q1), self.l2*cos(q2+q1)]
		# ])


		Jacobian=Matrix([[diff(x,q1),diff(x,q2)],[diff(y,q1),diff(y,q2)]])
		pos_ee=Matrix([x,y])
		vel_ee=Jacobian*Matrix([q1_dot,q2_dot])
		delta_x = Matrix([pos_ee, vel_ee]) - Matrix(self.xg)
	
		cost=0.5*delta_x.T*Q*delta_x
 
		if control:
			cost += 0.5 * Matrix([u1,u2]).T * R * Matrix([u1,u2])
		
		return cost
	
	def symbolic_cost_eval(self, control=True):


		q1,q2,q1_dot,q2_dot, u1, u2  = symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)

		# x=self.l2*cos(q2+q1)+self.l1*cos(q1)
		# y=self.l2*sin(q2+q1)+self.l1*sin(q1)

		x=-self.l2*sin(q2+q1)-self.l1*sin(q1)
		y=self.l2*cos(q2+q1)+self.l1*cos(q1)

		
		# J = Matrix([
		# 	[-self.l2*sin(q2+q1)-self.l1*sin(q1), -self.l2*sin(q2+q1)],
		# 	[self.l2*cos(q2+q1)-self.l1*cos(q1), self.l2*cos(q2+q1)]
		# ])

		Jacobian=Matrix([[diff(x,q1),diff(x,q2)],[diff(y,q1),diff(y,q2)]])
		pos_ee=Matrix([x,y])
		vel_ee=Jacobian@Matrix([q1_dot,q2_dot])
		state= Matrix([pos_ee, vel_ee])
		
		delta_x = state - Matrix(self.xg)

		cost=0.5*delta_x.T@Q@delta_x

		
		
		if control:
			
			cost += 0.5 * Matrix([u1,u2]).T @R @ Matrix([u1,u2])
			# to_return=lambdify([q1,q2,q1_dot ,q2_dot],state, "numpy"), lambdify([q1,q2,q1_dot ,q2_dot, u1, u2, Q, R, xg],cost, "numpy")
			return lambdify([q1,q2,q1_dot ,q2_dot,Q, R, u1, u2,xg],cost, "numpy")
			return to_return
		else:
			# return lambdify([q1,q2,q1_dot ,q2_dot],state, "numpy"), lambdify([q1,q2,q1_dot ,q2_dot, Q, xg],cost, "numpy")
			return lambdify([q1,q2,q1_dot ,q2_dot, Q, xg],cost, "numpy")

		

	# def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):

	# 	currQ = self.get_currQ(u,timestep)
	# 	if u is None:
	# 		cost_value=cost_control_off(x[0],x[1],x[2],x[3],currQ,self.xg)
	# 	else:
	# 		cost_value=cost_control_in(x[0],x[1],x[2],x[3],u,currQ,self.R,self.xg)

	# 	return cost_value
		
	
	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):

		currQ = self.get_currQ(u,timestep)
		if u is None:
			cost_value=self.cost_control_off_eval(x[0],x[1],x[2],x[3],currQ,self.xg)
		else:
			cost_value=self.cost_control_in_eval(x[0],x[1],x[2],x[3],currQ,self.R,u[0],u[1],self.xg)

		return cost_value
		


	def symbolic_gradient(self,control=True):
		q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')

		if(control):
			cost=self.cost_control_in
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot), diff(cost, u1), diff(cost, u2)]))
			
			return gradient
		else:
			
			cost=self.cost_control_off
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot)]))
			
			return gradient
		
	def symbolic_gradient_eval(self,control=True):
		q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)
		if(control):
			cost=self.cost_control_in
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot), diff(cost, u1), diff(cost, u2)]))
			to_return= lambdify([q1,q2,q1_dot ,q2_dot, Q,R,u1,u2],gradient, "numpy")
			return to_return
		else:
			cost=self.cost_control_off
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot)]))
			return  lambdify([q1,q2,q1_dot ,q2_dot, Q],gradient, "numpy")

	def gradient(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):

		# q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
		# Q= MatrixSymbol('Q',4,4)
		currQ = self.get_currQ(u,timestep)
	
		if u is None:
			symbolic_grad=self.grad_control_off_eval
			gradient_val= symbolic_grad(x[0], x[1],x[2], x[3],currQ)
		else:
			symbolic_grad=self.grad_control_in_eval
			gradient_val= symbolic_grad(x[0], x[1],x[2], x[3],currQ, self.R, u[0], u[1])



		# x_values = np.array([0.1, 0.2, 0.3, 0.4])  # Example values for q1, q2, q1_dot, q2_dot
		# Q_values = np.eye(4)  # Example value for Q matrix
		# R_values = np.eye(2)  # Example value for R matrix
		# u_values = np.array([0.5, 0.6])  # Example values for u1, u2
		# xg_values = np.array([[0.1], [0.2], [0.3], [0.4]])  # Example value for xg

		# Evaluate the lambdify expression
		# symbolic_grad=self.grad_control_in_eval
		# result = symbolic_grad(x_values[0], x_values[1], x_values[2], x_values[3], Q_values, R_values, u_values[0], u_values[1])

		# print(result)
		# return result


		return gradient_val
		
	# def gradient(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):

	# 	# q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
	# 	# Q= MatrixSymbol('Q',4,4)
	# 	currQ = self.get_currQ(u,timestep)

	# 	if u is None:
	# 		gradient_val=gradient_cost_off(x[0],x[1],x[2],x[3],currQ,self.xg)
	# 	else:
	# 		gradient_val=gradient_cost_in(x[0],x[1],x[2],x[3],u[0],u[1],currQ,self.R,self.xg)


	# 	return gradient_val



	def symbolic_hessian(self, control=True):
		q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
		if(control):
			cost=self.cost_control_in
			hessian = Matrix(BlockMatrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, q1_dot, q2_dot, u1, u2]] for q2_ in [q1, q2, q1_dot, q2_dot, u1, u2]]))
			
			return hessian
		else:
			cost=self.cost_control_off
			hessian= Matrix(BlockMatrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, q1_dot, q2_dot]] for q2_ in [q1, q2, q1_dot, q2_dot]]))
			return hessian
		
	def symbolic_hessian_eval(self, control=True):
		q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)
		if(control):
			cost=self.cost_control_in
			hessian = Matrix(BlockMatrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, q1_dot, q2_dot, u1, u2]] for q2_ in [q1, q2, q1_dot, q2_dot, u1, u2]]))
			return lambdify([q1,q2,q1_dot ,q2_dot, u1, u2, Q, R, xg],hessian, "numpy")
		else:
			cost=self.cost_control_off
			hessian= Matrix(BlockMatrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, q1_dot, q2_dot]] for q2_ in [q1, q2, q1_dot, q2_dot]]))
			return lambdify([q1,q2,q1_dot ,q2_dot, Q, xg],hessian, "numpy")

	# def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
	# 	# q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
	# 	# Q= MatrixSymbol('Q',4,4)
	# 	currQ = self.get_currQ(u,timestep)
	# 	hessian=self.hess_control_in
	# 	if u is None:
	# 		symbolic_hess=self.hess_control_off_eval
	# 		#hessian_evaluated = np.array(hessian.subs({q1: x[0], q2: x[1], q1_dot: x[2], q2_dot: x[3], Q:Matrix(currQ)})).astype(float)
	# 		hessian_val= symbolic_hess(x[0], x[1],x[2], x[3],currQ, self.xg)

	# 	else:
	# 		symbolic_hess=self.hess_control_in_eval
	# 		#hessian_evaluated =np.array(hessian.subs({q1: x[0], q2: x[1],q1_dot: x[2], q2_dot: x[3], u1: u[0], u2: u[1], Q:Matrix(currQ)})).astype(float)
	# 		print("xg", self.xg)
	# 		print("R", self.R)
	# 		print("Q", currQ)
	# 		hessian_val= symbolic_hess(x[0], x[1],x[2], x[3], u[0], u[1],currQ, self.R, self.xg)

	# 	return hessian_val
	
	# def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
	# 	currQ = self.get_currQ(u,timestep)
	# 	if (self.simplified_hess):
	# 		return self.gradient.transpose@self.gradient
	# 	else:
	# 		if u is None:
	# 			hessian_eval=hessian_cost_off(x[0],x[1],x[2],x[3],currQ,self.xg)

	# 		else:
	# 			hessian_eval=hessian_cost_in(x[0],x[1],x[2],x[3],u[0],u[1],currQ,self.R,self.xg)
	# 		return hessian_eval
		
	def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		currQ = self.get_currQ(u,timestep)
		
		if u is None:
			hessian_eval=hessian_cost_off(x[0],x[1],x[2],x[3],currQ,self.xg)

		else:
			hessian_eval=hessian_cost_in(x[0],x[1],x[2],x[3],u[0],u[1],currQ,self.R,self.xg)
		return hessian_eval

			
		




	def get_currQ(self, u = None, timestep = None):
		last_state = isinstance(u,type(None))
		shifted_QF = (not isinstance(timestep,type(None)) \
			          and not isinstance(self.QF_start,type(None)) \
			          and timestep >= self.QF_start)
		use_QF =  last_state or shifted_QF
		currQ = self.QF if use_QF else self.Q
		return currQ
		


	def increase_QF(self, multiplier: float = 2.0):
		self.QF *= multiplier
		self.increaseCount_QF += 1
		return self.increaseCount_QF

	def increase_Q(self, multiplier: float = 2.0):
		self.Q *= multiplier
		self.increaseCount_Q += 1
		return self.increaseCount_Q

	def reset_increase_count_QF(self):
		self.increaseCount_QF = 0

	def reset_increase_count_Q(self):
		self.increaseCount_Q = 0

	def shift_QF_start(self, shift: float = -1.0):
		self.QF_start += shift
		self.QF_start = max(self.QF_start, 0)
		return self.QF_start


class UrdfCost(TrajoptCost):

	def __init__(self,  plant: URDFPlant , Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None):
		
		self.plant=plant
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start
		self.l1=1
		self.l2=1
		self.simplified_hess=False

	def symbolic_vel(self):
		#recupérer symbolic ee_pos_grad[:2] from rbd (=J)
		J=self.plant.rbdReference.end_effector_position_gradients(x[:2],offsets = [np.matrix([[0,1,0,1]])])[0][:2]
		q1_dot,q2_dot=symbols(q1_dot,q2_dot)
		vel=J@Matrix([q1_dot,q2_dot])
		return vel


	def compute_J(self,q):

		# if(rnea):
		# 	_,vel,_,_=self.plant.rbdReference.rnea(q,qd)
		# else:
		# 	[q1,q2]=q
		# 	J = np.array([
		# 			[-self.l2*np.sin(q2+q1)-self.l1*np.sin(q1), -self.l2*np.sin(q2+q1)],
		# 			[self.l2*np.cos(q2+q1)-self.l1*np.cos(q1), self.l2*np.cos(q2+q1)]
		# 		])
		J=self.plant.rbdReference.end_effector_position_gradients(q,offsets = [np.matrix([[0,1,0,1]])])[0][:2,:2]
			#vel=J@qd
		
		return J
			

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):

		# pos = self.plant.rbdReference.end_effector_positions(x[:2],offsets = [np.matrix([[0,1,0,1]])])[0][:2]
		# vel = (self.compute_J(x[:2])@x[2:4]).transpose()
		# X = np.vstack((pos,vel))
		# delta_x = X - self.xg

		dx=self.delta_x(x)
		currQ = self.get_currQ(u,timestep)
		cost = 0.5*np.matmul(dx.transpose(),np.matmul(currQ,dx))
		if not isinstance(u, type(None)):
			cost += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		return cost
	
	def delta_x(self, x: np.ndarray):
		pos = self.plant.rbdReference.end_effector_positions(x[:2],offsets = [np.matrix([[0,1,0,1]])])[0][:2]
		vel = (self.compute_J(x[:2])@x[2:4]).transpose()
		X = np.array(np.vstack((pos,vel))).reshape(4,)
		return X - self.xg
		
	def gradient(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		dx=self.delta_x(x)
		currQ = self.get_currQ(u,timestep)
		J_tot=self.plant.rbdReference.jacobian_tot_state(x[:2],x[2:])
		top=np.array(dx.transpose()@currQ@J_tot).reshape(4,)



		# dee_pos=self.compute_J(x[:2]).reshape(4,1)
		# dee_vel=np.zeros((4,1))
		
		# b = np.matmul(currQ,delta_x)
		# top = np.vstack((dee_pos,dee_vel))*b

		if u is None:
			grad= top
		else:
			bottom = (np.matmul(u.transpose(),self.R))
			grad= np.hstack((top,bottom))
		
		return grad

		

		

	def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		grad=self.gradient(x,u)
		n=grad.shape[0]
		hess=grad.reshape(n,1)@(grad.reshape(1,n))
		return hess
		
	def get_currQ(self, u = None, timestep = None):

		last_state = isinstance(u,type(None))
		shifted_QF = (not isinstance(timestep,type(None)) \
			          and not isinstance(self.QF_start,type(None)) \
			          and timestep >= self.QF_start)
		use_QF =  last_state or shifted_QF
		currQ = self.QF if use_QF else self.Q
		return currQ
	
	def increase_QF(self, multiplier: float = 2.0):
		self.QF *= multiplier
		self.increaseCount_QF += 1
		return self.increaseCount_QF

	def increase_Q(self, multiplier: float = 2.0):
		self.Q *= multiplier
		self.increaseCount_Q += 1
		return self.increaseCount_Q

	def reset_increase_count_QF(self):
		self.increaseCount_QF = 0

	def reset_increase_count_Q(self):
		self.increaseCount_Q = 0

	def shift_QF_start(self, shift: float = -1.0):
		self.QF_start += shift
		self.QF_start = max(self.QF_start, 0)
		return self.QF_start
	


class NumericalCost(TrajoptCost):

	def __init__(self, Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None):
		
		
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start
		self.l1=1
		self.l2=1

	def numerical_gradient(self, f, x,u, h=1e-5):

		state=np.concatenate([x, u])

		grad = np.zeros_like(state)
		for i in range(len(state)):
			old_value = state[i]
			
			# Compute f(x + h)
			state[i] = old_value + h
			fxh1 = f(state[:4],state[4:])
			
			# Compute f(x - h)
			state[i] = old_value - h
			fxh2 = f(state[:4],state[4:])
			
			# Compute the gradient component
			grad[i] = (fxh1 - fxh2) / (2 * h)
			
			# Reset x[i] to its original value
			state[i] = old_value
		
		return grad
	
	def cost_function(self, x: np.ndarray, u: np.ndarray):
		
		currQ = self.Q
		
		
		J=np.zeros((2,2))
		J[0,0]=-self.l1*np.cos(x[0])-self.l2*np.cos(x[0]+x[1])
		J[0,1]=-self.l2*np.cos(x[0]+x[1])
		J[1,0]=-self.l1*np.sin(x[0])-self.l2*np.sin(x[0]+x[1])
		J[1,1]=-self.l2*np.sin(x[0]+x[1])

		v=J@x[2:4] #q_dot
		pos=np.array([-self.l2*np.sin(x[0]+x[1])-self.l1*np.sin(x[0]),\
		 		self.l2*np.cos(x[0]+x[1])+self.l1*np.cos(x[0])])
		dx = np.concatenate((pos, v)) - self.xg									  
		cost_func= 0.5*np.matmul(dx.transpose(),np.matmul(currQ,dx))
		if not isinstance(u, type(None)):
			cost_func += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		return cost_func

	
		
	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		return self.cost_function(x,u)
	
	def gradient(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		return self.numerical_gradient(self.cost_function, x, u) # control off for now
	
	def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		grad=self.gradient(x,u)
		return grad@(grad.transpose())
	
	def get_currQ(self, u = None, timestep = None):

		last_state = isinstance(u,type(None))
		shifted_QF = (not isinstance(timestep,type(None)) \
			          and not isinstance(self.QF_start,type(None)) \
			          and timestep >= self.QF_start)
		use_QF =  last_state or shifted_QF
		currQ = self.QF if use_QF else self.Q
		return currQ
	
	def increase_QF(self, multiplier: float = 2.0):
		self.QF *= multiplier
		self.increaseCount_QF += 1
		return self.increaseCount_QF

	def increase_Q(self, multiplier: float = 2.0):
		self.Q *= multiplier
		self.increaseCount_Q += 1
		return self.increaseCount_Q

	def reset_increase_count_QF(self):
		self.increaseCount_QF = 0

	def reset_increase_count_Q(self):
		self.increaseCount_Q = 0

	def shift_QF_start(self, shift: float = -1.0):
		self.QF_start += shift
		self.QF_start = max(self.QF_start, 0)
		return self.QF_start
	

