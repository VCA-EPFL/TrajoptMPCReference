import numpy as np
from sympy import symbols, diff, Matrix, cos, sin, MatrixSymbol, BlockMatrix, lambdify


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
		# print('get_currQ')
		last_state = isinstance(u,type(None))
		shifted_QF = (not isinstance(timestep,type(None)) \
			          and not isinstance(self.QF_start,type(None)) \
			          and timestep >= self.QF_start)
		use_QF =  last_state or shifted_QF
		currQ = self.QF if use_QF else self.Q
		return currQ

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		# print('value')
		
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
		_,self.cost_control_in_eval=self.symbolic_cost_eval(control=True)
		self.cost_control_off=self.symbolic_cost(control=False)
		_,self.cost_control_off_eval=self.symbolic_cost_eval(control=False)
		self.grad_control_in=self.symbolic_gradient(control=True)
		self.grad_control_off=self.symbolic_gradient(control=False)
		self.grad_control_in_eval=self.symbolic_gradient_eval(control=True)
		self.grad_control_off_eval=self.symbolic_gradient_eval(control=False)
		self.hess_control_in=self.symbolic_hessian(control=True)
		self.hess_control_off=self.symbolic_hessian(control=False)
		self.hess_control_in_eval=self.symbolic_hessian_eval(control=True)
		self.hess_control_off_eval=self.symbolic_hessian_eval(control=False)




	def symbolic_cost(self, control=True):

		q1,q2,q1_dot,q2_dot, u1, u2  = symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		# x=(self.l2*cos(q2)+self.l1)*cos(q1)
		# y=(self.l2*cos(q2)+self.l1)*sin(q1)

		x=self.l2*cos(q2+q1)+self.l1*cos(q1)
		y=self.l2*sin(q2+q1)+self.l1*sin(q1)

		
		J = Matrix([
			[-self.l2*sin(q2+q1)-self.l1*sin(q1), -self.l2*sin(q2+q1)],
			[self.l2*cos(q2+q1)-self.l1*cos(q1), self.l2*cos(q2+q1)]
		])

		# x=self.l2*cos(q2+q1 + np.pi)+self.l1*cos(q1+np.pi/2)
		# y=self.l2*sin(q2+q1+ np.pi)+self.l1*sin(q1+ np.pi/2)

		
		# J = Matrix([
		# 	[-self.l2*sin(q2+q1+ np.pi)-self.l1*sin(q1+ np.pi/2), -self.l2*sin(q2+q1+ np.pi)],
		# 	[self.l2*cos(q2+q1+ np.pi)-self.l1*cos(q1+ np.pi/2), self.l2*cos(q2+q1+ np.pi)]
		# ])
		Jacobian=Matrix([[diff(x,q1),diff(x,q2)],[diff(y,q1),diff(y,q2)]])
		pos_ee=Matrix([x,y])
		vel_ee=Jacobian@Matrix([q1_dot,q2_dot])
		delta_x = Matrix([pos_ee, vel_ee]) - Matrix(self.xg)
	
		cost=0.5*delta_x.T@Q@delta_x
 
		if control:
			cost += 0.5 * Matrix([u1,u2]).T * self.R * Matrix([u1,u2])

		return cost
	
	def symbolic_cost_eval(self, control=True):

		q1,q2,q1_dot,q2_dot, u1, u2  = symbols('q1 q2 q1_dot q2_dot u1 u2')
		Q= MatrixSymbol('Q',4,4)
		# x=(self.l2*cos(q2)+self.l1)*cos(q1)
		# y=(self.l2*cos(q2)+self.l1)*sin(q1)

		x=self.l2*cos(q2+q1)+self.l1*cos(q1)
		y=self.l2*sin(q2+q1)+self.l1*sin(q1)

		
		J = Matrix([
			[-self.l2*sin(q2+q1)-self.l1*sin(q1), -self.l2*sin(q2+q1)],
			[self.l2*cos(q2+q1)-self.l1*cos(q1), self.l2*cos(q2+q1)]
		])

		Jacobian=Matrix([[diff(x,q1),diff(x,q2)],[diff(y,q1),diff(y,q2)]])
		pos_ee=Matrix([x,y])
		vel_ee=Jacobian@Matrix([q1_dot,q2_dot])
		state= Matrix([pos_ee, vel_ee])
		
		delta_x = state - Matrix(self.xg)
		cost=0.5*delta_x.T@Q@delta_x

		
		
		if control:
			cost += 0.5 * Matrix([u1,u2]).T @ self.R @ Matrix([u1,u2])
			return lambdify([q1,q2,q1_dot ,q2_dot],state, "numpy"), lambdify([q1,q2,q1_dot ,q2_dot, u1, u2, Q],cost, "numpy")
		
		else:
			return lambdify([q1,q2,q1_dot ,q2_dot],state, "numpy"), lambdify([q1,q2,q1_dot ,q2_dot, Q],cost, "numpy")

		

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		
		currQ = self.get_currQ(u,timestep)

		if u is None:
			# print("control off")
			symbolic_cost=self.cost_control_off_eval
			cost_value= symbolic_cost(x[0], x[1],x[2], x[3],currQ)
			#cost_value= symbolic_cost.subs({q1: x[0], q2: x[1],q1_dot: x[2], q2_dot: x[3], Q:Matrix(currQ)})[0,0]
		else:
			symbolic_cost=self.cost_control_in_eval
			cost_value= symbolic_cost(x[0], x[1],x[2], x[3], u[0], u[1],currQ)
		
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
		if(control):
			cost=self.cost_control_in
			gradient = Matrix(BlockMatrix([diff(cost, q1), diff(cost, q2), diff(cost, q1_dot), diff(cost, q2_dot), diff(cost, u1), diff(cost, u2)]))
			return lambdify([q1,q2,q1_dot ,q2_dot,u1, u2, Q],gradient, "numpy")

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
			#gradient_val=np.array(symbolic_grad.subs({q1: x[0], q2: x[1],q1_dot: x[2], q2_dot: x[3], Q:Matrix(currQ)})).astype(float)
			gradient_val= symbolic_grad(x[0], x[1],x[2], x[3],currQ)
		else:
			symbolic_grad=self.grad_control_in_eval
			#gradient_val=np.array(symbolic_grad.subs({q1: x[0], q2: x[1],q1_dot: x[2], q2_dot: x[3], u1: u[0], u2: u[1], Q:Matrix(currQ)})).astype(float)
			gradient_val= symbolic_grad(x[0], x[1],x[2], x[3], u[0], u[1],currQ)

		return gradient_val



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
		if(control):
			cost=self.cost_control_in
			hessian = Matrix(BlockMatrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, q1_dot, q2_dot, u1, u2]] for q2_ in [q1, q2, q1_dot, q2_dot, u1, u2]]))
			return lambdify([q1,q2,q1_dot ,q2_dot, u1, u2, Q],hessian, "numpy")
		else:
			cost=self.cost_control_off
			hessian= Matrix(BlockMatrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, q1_dot, q2_dot]] for q2_ in [q1, q2, q1_dot, q2_dot]]))
			return lambdify([q1,q2,q1_dot ,q2_dot, Q],hessian, "numpy")

	def hessian(self,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		# q1, q2, q1_dot, q2_dot, u1, u2= symbols('q1 q2 q1_dot q2_dot u1 u2')
		# Q= MatrixSymbol('Q',4,4)
		currQ = self.get_currQ(u,timestep)
		hessian=self.hess_control_in
		if u is None:
			symbolic_hess=self.hess_control_off_eval
			#hessian_evaluated = np.array(hessian.subs({q1: x[0], q2: x[1], q1_dot: x[2], q2_dot: x[3], Q:Matrix(currQ)})).astype(float)
			hessian_val= symbolic_hess(x[0], x[1],x[2], x[3],currQ)

		else:
			symbolic_hess=self.hess_control_in_eval
			#hessian_evaluated =np.array(hessian.subs({q1: x[0], q2: x[1],q1_dot: x[2], q2_dot: x[3], u1: u[0], u2: u[1], Q:Matrix(currQ)})).astype(float)
			hessian_val= symbolic_hess(x[0], x[1],x[2], x[3], u[0], u[1],currQ)

		return hessian_val




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
