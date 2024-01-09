import numpy as np
from sympy import symbols, diff, Matrix, cos, sin

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
		self.l2=2

	def value(self, x: np.ndarray, u: np.ndarray = None, timestep: int = None):

		[q1,q2] = x[:2]
		currQ = self.get_currQ(u,timestep)
		pos_ee=[(self.l2*cos(q2)+self.l1)*cos(q1),(self.l2*cos(q2)+self.l1)*sin(q1)]
		delta_x = pos_ee - self.xg
		cost = 0.5*np.matmul(delta_x.transpose(),np.matmul(currQ,delta_x))
		if not isinstance(u, type(None)):
			cost += 0.5*np.matmul(u.transpose(),np.matmul(self.R,u))
		return cost
	

	def gradient(self,l1,l2,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		q1,q2,u1,u2 = symbols('q1 q2 u1 u2')
		currQ = self.get_currQ(u,timestep)
		cost=self.symbolic_cost(q1,q2,u1,u2,currQ)
		if u is None:
			gradient = Matrix([diff(cost, q1), diff(cost, q2), diff(cost, u1), diff(cost, u2)])
			return gradient.subs({q1: x[0], q2: x[1]})
		else:
			gradient = Matrix([diff(cost, q1), diff(cost, q2), diff(cost, u1), diff(cost, u2)])
			return gradient.subs({q1: x[0], q2: x[1],q1: u[0], q2: u[1]})
		
		
	
	def hessian(self,l1,l2,x: np.ndarray, u: np.ndarray = None, timestep: int = None):
		q1,q2,u1,u2 = symbols('q1 q2 u1 u2')
		currQ = self.get_currQ(u,timestep)
		cost=self.symbolic_cost(q1,q2,u1,u2, currQ)

		if u is None:
			hessian = Matrix([[diff(cost, q1, q1), diff(cost, q1, q2)],
                          [diff(cost, q2, q1), diff(cost, q2, q2)]])
			return hessian.subs({q1: x[0], q2: x[1]})
		else:
			hessian = Matrix([[diff(cost, q1_, q2_) for q1_ in [q1, q2, u1, u2]] for q2_ in [q1, q2, u1, u2]])
			return hessian.subs({q1: x[0], q2: x[1],q1: u[0], q2: u[1]})
		

	def symbolic_cost(self,q1,q2,u1,u2,Q):
		pos_ee=[(self.l2*cos(q2)+self.l1)*cos(q1),(self.l2*cos(q2)+self.l1)*sin(q1)]
		delta_x = Matrix(pos_ee) - Matrix(self.xg)
		cost=0.5*delta_x.T*Q*delta_x +0.5 * Matrix([u1,u2]).T * self.R * Matrix([u1,u2]) 
		return cost


	
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
