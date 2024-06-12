import numpy as np
from sympy import symbols, diff, Matrix, cos, sin, MatrixSymbol, BlockMatrix, lambdify, ccode
from expressions import *
from TrajoptPlant import *
import csv





class TrajoptCost:
	def value():
		raise NotImplementedError

	def gradient():
		raise NotImplementedError

	def hessian():
		raise NotImplementedError

#------------------------------------------------------------QUADRATIC COST ----------------------------------------------------------------------------------------------------------------

class QuadraticCost(TrajoptCost):
	def __init__(self, Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None):
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start
		self.saved_cost=[]
		self.saved_grad=[]
		self.saved_hess=[]
		

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
		self.saved_cost.append([cost])
		return cost

	def gradient(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		delta_x = x - self.xg
		currQ = self.get_currQ(u,timestep)
		top = np.matmul(delta_x.transpose(),currQ)

		if u is None:
			grad= top
		else:
			bottom = np.matmul(u.transpose(),self.R)
			grad = np.hstack((top,bottom))
		self.saved_grad.append(grad)
		return grad

	def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		nx = self.Q.shape[0]
		nu = self.R.shape[0]
		currQ = self.get_currQ(u,timestep)

		if u is None:
			hess= currQ
		else:
			top = np.hstack((currQ,np.zeros((nx,nu))))
			bottom = np.hstack((np.zeros((nu,nx)),self.R))
			hess= np.vstack((top,bottom))
		self.saved_hess.append(hess)
		return hess

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



#-------------------------------------------------------------SYMBOLIC COST ----------------------------------------------------------------------------------------------------------------


class ArmCost(TrajoptCost):

	def __init__(self, Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, simplified_hessian: bool,QF_start = None):
		
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start
		self.l1=1
		self.l2=1
		self.simplified_hess=simplified_hessian
		self.cost_control_in=self.symbolic_cost(control=True)
		self.cost_control_in_eval=self.symbolic_cost_eval(control=True)
		self.cost_control_off=self.symbolic_cost(control=False)
		self.cost_control_off_eval=self.symbolic_cost_eval(control=False)
		self.grad_control_in=self.symbolic_gradient(control=True)
		self.grad_control_off=self.symbolic_gradient(control=False)
		self.grad_control_in_eval=self.symbolic_gradient_eval(control=True)
		self.grad_control_off_eval=self.symbolic_gradient_eval(control=False)
		if not self.simplified_hess:
			self.hess_control_in=self.symbolic_hessian(control=True)
			self.hess_control_off=self.symbolic_hessian(control=False)
			self.hess_control_in_eval=self.symbolic_hessian_eval(control=True)
			self.hess_control_off_eval=self.symbolic_hessian_eval(control=False)		
		self.saved_cost=[]
		self.saved_grad=[]
		self.saved_hess=[]
		self.saved_simple_hess=[]

	# Is used by the gradient to find the symbolic derivative
	def symbolic_cost(self, control=True):

		q1,q2,q1_dot,q2_dot, u1, u2  = symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)
		x=-self.l2*sin(q2+q1)-self.l1*sin(q1)
		y=self.l2*cos(q2+q1)+self.l1*cos(q1)
		c12 = cos(q2+q1)
		s12 = sin(q2+q1)
		J = Matrix([
			[-self.l2*c12-self.l1*cos(q1), -self.l2*c12],
			[-self.l2*s12-self.l1*sin(q1), -self.l2*s12]
		])
		#Jacobian=Matrix([[diff(x,q1),diff(x,q2)],[diff(y,q1),diff(y,q2)]])
		pos_ee=Matrix([x,y])
		vel_ee=J*Matrix([q1_dot,q2_dot])
		delta_x = Matrix([pos_ee, vel_ee]) - Matrix(self.xg)
		cost=0.5*delta_x.T*Q*delta_x
		if control:
			cost += 0.5 * Matrix([u1,u2]).T * R * Matrix([u1,u2])
		return cost
	
	# Is computed once, cost that can be evaluated online to find the value
	def symbolic_cost_eval(self, control=True):

		q1,q2,q1_dot,q2_dot, u1, u2  = symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)
		x=-self.l2*sin(q2+q1)-self.l1*sin(q1)
		y=self.l2*cos(q2+q1)+self.l1*cos(q1)
		c12 = cos(q2+q1)
		s12 = sin(q2+q1)
		J = Matrix([
			[-self.l2*c12-self.l1*cos(q1), -self.l2*c12],
			[-self.l2*s12-self.l1*sin(q1), -self.l2*s12]
		])
		#Jacobian=Matrix([[diff(x,q1),diff(x,q2)],[diff(y,q1),diff(y,q2)]])
		pos_ee=Matrix([x,y])
		vel_ee=J@Matrix([q1_dot,q2_dot])
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

		
					# This was used when all the expressions of the cost were in an external file
					# def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
					# 	currQ = self.get_currQ(u,timestep)
					# 	if u is None:
					# 		cost_value=cost_control_off(x[0],x[1],x[2],x[3],currQ,self.xg)
					# 	else:
					# 		cost_value=cost_control_in(x[0],x[1],x[2],x[3],u,currQ,self.R,self.xg)
					# 	return cost_value
	
	def current_state(self,x: np.ndarray):
		[q1,q2,q1_d,q2_d]=x
		c12 = np.cos(q2+q1)
		s12 = np.sin(q2+q1)
		J = Matrix([
			[-self.l2*c12-self.l1*np.cos(q1), -self.l2*c12],
			[-self.l2*s12-self.l1*np.sin(q1), -self.l2*s12]
		])
		v=J@x[2:4]
		x= np.array([-self.l2*np.sin(q2+q1)-self.l1*np.sin(q1),\
					self.l2*np.cos(q2+q1)+self.l1*np.cos(q1)])
		return np.concatenate((x,v))

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):

		currQ = self.get_currQ(u,timestep)
		if u is None:
			cost_value=self.cost_control_off_eval(x[0],x[1],x[2],x[3],currQ,self.xg)
		else:
			cost_value=self.cost_control_in_eval(x[0],x[1],x[2],x[3],currQ,self.R,u[0],u[1],self.xg)
		self.saved_cost.append([cost_value[0][0]])
		return cost_value[0][0]
		
	# Is used by the hessian to find the symbolic derivative of the gradient
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
		
	# Evaluated version, computed once, evaluated online
	def symbolic_gradient_eval(self,control=True):
		q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		R= MatrixSymbol('R',2,2)
		xg=MatrixSymbol('xg',4,1)
		if(control):
			cost=self.cost_control_in #symbolic expression
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot), diff(cost, u1), diff(cost, u2)]))
			to_return= lambdify([q1,q2,q1_dot ,q2_dot, Q,R,u1,u2],gradient, "numpy")
			return to_return
		else:
			cost=self.cost_control_off
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot)]))
			return  lambdify([q1,q2,q1_dot ,q2_dot, Q],gradient, "numpy")

	def gradient(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		currQ = self.get_currQ(u,timestep)
		if u is None:
			symbolic_grad=self.grad_control_off_eval #ready to evaluate expression
			gradient_val= symbolic_grad(x[0], x[1],x[2], x[3],currQ)
		else:
			symbolic_grad=self.grad_control_in_eval
			gradient_val= symbolic_grad(x[0], x[1],x[2], x[3],currQ, self.R, u[0], u[1])
		self.saved_grad.append(gradient_val)
		return gradient_val[0]
		
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
			cost=self.cost_control_in #symbolic version
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

	def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		currQ = self.get_currQ(u,timestep)
		if (self.simplified_hess):
			grad= self.gradient(x,u)
			n=grad.shape[0]
			grad=grad.reshape((n,1))
			simplified_hess=grad@grad.transpose()
			self.saved_simple_hess.append(simplified_hess)
		else:
			hessian=self.hess_control_in
			if u is None:
				symbolic_hess=self.hess_control_off_eval #ready to evaluate cost
				hessian_val= symbolic_hess(x[0], x[1],x[2], x[3],currQ, self.xg)
			else:
				symbolic_hess=self.hess_control_in_eval
				hessian_val= symbolic_hess(x[0], x[1],x[2], x[3], u[0], u[1],currQ, self.R, self.xg)
			self.saved_hess.append(hessian_val)
		return hessian_val
		
					# def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
					# 	currQ = self.get_currQ(u,timestep)
					# 	if u is None:
					# 		hessian_eval=hessian_cost_off(x[0],x[1],x[2],x[3],currQ,self.xg)
					# 	else:
					# 		hessian_eval=hessian_cost_in(x[0],x[1],x[2],x[3],u[0],u[1],currQ,self.R,self.xg)
					# 	return hessian_eval

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




#-------------------------------------------------------------------URDF COST ----------------------------------------------------------------------------------------------------------------
	
# For now works only with numpy, no overloading
class UrdfCost(TrajoptCost):

	def __init__(self,  plant: URDFPlant , Q_in: np.ndarray, QF_in: np.ndarray, R_in: np.ndarray, xg_in: np.ndarray, QF_start = None, overloading=False):
		self.plant=plant
		self.Q = Q_in
		self.QF = QF_in
		self.R = R_in
		self.xg = xg_in
		self.increaseCount_Q = 0
		self.increaseCount_QF = 0
		self.QF_start = QF_start
		self.saved_cost=[]
		self.saved_grad=[]
		self.saved_hess=[]
		self.n=self.plant.get_num_pos() # n joints
		self.offsets=[np.matrix([[0,1,0,1]])] # May need to be updated if change in URDF
		self.plant.rbdReference.overloading= overloading
		self.overloading=overloading

	def compute_J(self,q): # online value of the Jacobian
		J=self.plant.rbdReference.Jacobian(q,offsets = self.offsets)
		return J
			
	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		dx=self.delta_x(x)
		currQ = self.get_currQ(u,timestep)
		cost = 0.5*np.matmul(dx.transpose(),np.matmul(currQ,dx))
		if not isinstance(u, type(None)):
			cost += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		self.saved_cost.append([cost]) 
		if(self.overloading):
			return np.float64(cost)
		else:
			return cost
	
	def delta_x(self, x: np.ndarray):
		pos = self.plant.rbdReference.end_effector_positions(x[:self.n],self.offsets)
		if(self.overloading):
			vel = (self.compute_J(x[:self.n])@x[self.n:]).reshape(self.n,1)# v=J*qd
			# print("Pos\n", pos)
			# print("vel\n", vel)
			X = matrix_((np.vstack((pos,vel))).reshape(2*self.n,))
		else:
			vel = (self.compute_J(x[:self.n])@x[self.n:]).transpose() # v=J*qd
			X = np.array(np.vstack((pos,vel))).reshape(2*self.n,)
			# print("Pos\n", pos)
			# print("vel\n", vel)

		return X - self.xg
		
	def gradient(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		dx=self.delta_x(x)
		currQ = self.get_currQ(u,timestep)
		J_tot=self.plant.rbdReference.jacobian_tot_state(x[:self.n],x[self.n:])
		if(self.overloading):
			top=(dx.transpose()@currQ@J_tot).reshape(2*self.n,)
			if u is None:
				grad= top
			else:
				bottom = (u.transpose()@self.R).reshape(self.n,)
				# print("bottom\n", bottom.shape)
				# print("u\n", u.transpose().shape)
				# print("R\n", self.R)
				grad= matrix_(np.hstack((top,bottom)))
		else:
			top=np.array(dx.transpose()@currQ@J_tot).reshape(2*self.n,)
			if u is None:
				grad= top
			else:
			
				bottom = (np.matmul(u.transpose(),self.R))

				grad= np.hstack((top,bottom))
		self.saved_grad.append(grad)
		return grad

	def dJtotdq(self, q: np.ndarray, qd: np.ndarray):
		dJdq=self.plant.rbdReference.dJdq(q, self.offsets)
		ddJdq = self.plant.rbdReference.d2Jdq2(q, self.offsets)
		n=self.n
		if(self.overloading):
			A=matrix_(np.hstack((dJdq, np.zeros((2*n,n)))).reshape(n, n, 2*n))
			B= matrix_(np.hstack((dJdq,ddJdq)).reshape(n, n, 2*n))
			dJtotdq = matrix_(np.zeros((2*n, 2*n, 2*n)))
			dJtotdq[0:n, 0:n, :] = A #top left
			dJtotdq[0:n, n:2*n, :] = matrix_(np.zeros((n, n, 2*n))) #top right
			dJtotdq[n:2*n, 0:n, :] = B #bottom left
			dJtotdq[n:2*n, n:2*n, :] = A #bottom right
		else:
			A=np.hstack((dJdq, np.zeros((2*n,n)))).reshape(n, n, 2*n)
			B= np.hstack((dJdq,ddJdq)).reshape(n, n, 2*n)
			dJtotdq = np.zeros((2*n, 2*n, 2*n))
			dJtotdq[0:n, 0:n, :] = A #top left
			dJtotdq[0:n, n:2*n, :] = np.zeros((n, n, 2*n)) #top right
			dJtotdq[n:2*n, 0:n, :] = B #bottom left
			dJtotdq[n:2*n, n:2*n, :] = A #bottom right
		return dJtotdq
	
	def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		q=x[:self.n]
		qd=x[self.n:]
		nx = self.Q.shape[0]
		nu = self.R.shape[0]
		Jtot=self.plant.rbdReference.jacobian_tot_state(q,qd, self.offsets)
		
		dJtotdq=self.dJtotdq(q,qd)
		currQ = self.get_currQ(u,timestep)
		dx=self.delta_x(x).reshape((2*self.n,1))
		hess1=((currQ@Jtot).T)@Jtot
		#hess2=(dx.transpose()@currQ@dJtotdq).reshape((4,4))
		hess_x=hess1 #+hess2
		# hess_x=np.zeros((4,4))
		if u is None:
			hess= hess_x
		else:
			if(self.overloading):
				top = matrix_(np.hstack((hess_x,np.zeros((nx,nu)))))
				bottom = matrix_(np.hstack((np.zeros((nu,nx)),self.R)))
				hess= matrix_(np.vstack((top,bottom)))
			else:
				top = np.hstack((hess_x,np.zeros((nx,nu))))
				bottom = np.hstack((np.zeros((nu,nx)),self.R))
				hess= np.vstack((top,bottom))
		self.saved_hess.append(hess)
		# n= 4 if (u is None) else 6
		# return np.zeros((n,n))
		return hess


	# SIMPLIFIED HESSIAN => DOESN'T WORK
	# def hessian(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
	# 	grad=self.gradient(x,u)
	# 	n=grad.shape[0]
	# 	hess=grad.reshape(n,1)@(grad.reshape(1,n))
	# 	self.saved_hess.append(hess)
	# 	return hess

	# HESSIAN FROM EVALUATION FILE => NOT UP TO DATE BC ROTATION AXIS IN THE URDF HAS CHANGED
	# def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
	# 	currQ = self.get_currQ(u,timestep)
	# 	if u is None:
	# 		hessian_eval=hessian_cost_off(x[0],x[1],x[2],x[3],currQ,self.xg)
	# 	else:
	# 		hessian_eval=hessian_cost_in(x[0],x[1],x[2],x[3],u[0],u[1],currQ,self.R,self.xg)	
	# 	self.saved_hess.append(hessian_eval)
	# 	return hessian_eval

		
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
			state[i] = old_value + h
			fxh1 = f(state[:4],state[4:])
			state[i] = old_value - h
			fxh2 = f(state[:4],state[4:])
			grad[i] = (fxh1 - fxh2) / (2 * h)
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
	

