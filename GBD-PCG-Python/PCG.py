import numpy as np
from overloading import matrix_

class PCG:
	def __init__(self, A, b, block_size, Nblocks, guess = None, options = {},overloading=False):
		self.A = A
		self.b = b
		self.block_size = block_size
		self.Nblocks = Nblocks
		self.guess = guess
		if self.guess == None:
			self.guess = np.zeros(self.A.shape[0])
		self.options = options
		self.overloading=overloading
		self.set_default_options(self.options)
		self.Pinv = self.compute_preconditioner(self.A, self.block_size, self.options['preconditioner_type'])
		

	def set_default_options(self, options):
		options.setdefault('exit_tolerance', 1e-6)
		options.setdefault('max_iter', 100)
		options.setdefault('DEBUG_MODE', False)
		options.setdefault('RETURN_TRACE', False)
		options.setdefault('preconditioner_type', 'BJ')
		self.validate_precon_type(options["preconditioner_type"])

	def update_A(self, A):
		self.A = A

	def update_b(self, b):
		self.b = b

	def update_guess(self, guess):
		self.guess = guess

	def update_exit_tolerance(self, tol):
		self.options["exit_tolerance"] = tol

	def update_max_iter(self, max_iter):
		self.options["max_iter"] = max_iter
	
	def update_preconditioner_type(self, type):
		self.validate_precon_type(type)
		self.options["preconditioner_type"] = type

	def update_DEBUG_MODE(self, mode):
		self.options["DEBUG_MODE"] = mode

	def update_RETURN_TRACE(self, mode):
		self.options["RETURN_TRACE"] = mode

	def validate_precon_type(self, precon_type):
		if not (precon_type in ['0', 'J', 'BJ', 'SS']):
			print("Invalid preconditioner options are [0: none, J : Jacobi, BJ: Block-Jacobi, SS: Symmetric Stair]")
			exit()

	# def invert_matrix(self, A):
	# 	print("invert_matrix")
	# 	try:
	# 		return np.linalg.inv(A)
	# 	except:
	# 		if self.options.get('DEBUG_MODE'):
	# 			print("Warning singular matrix -- using Psuedo Inverse.")
	# 		return np.linalg.pinv(A)

	def pcg(self, A, b, Pinv, guess, options = {}):
		self.set_default_options(options)
		trace = []
		
		if(self.overloading):
		# initialize
			x = matrix_.reshape(guess, (guess.shape[0],1))
		else:
			x = np.reshape(guess, (guess.shape[0],1))

		r = b-(A@x)
		r_tilde = Pinv@r
		p = r_tilde
		nu = r.transpose()@r_tilde
		# if options['DEBUG_MODE']:
		# 	print("		PCG: Initial nu[", nu, "]")
		trace = nu[0].tolist()
		trace2 = [np.linalg.norm(b - np.matmul(A, x))]
		# loop
		for iteration in range(options['max_iter']): #101
			
			Ap = A@p
			alpha = nu/(p.transpose()@Ap)
			r = r-Ap*alpha
			x = x+p*alpha
			r_tilde = Pinv@r
			nu_prime = r.transpose()@r_tilde

			trace.append(nu_prime.tolist()[0][0])
			trace2.append(np.linalg.norm(b - np.matmul(A, x)))

			if abs(nu_prime) < options['exit_tolerance']:
				if options['DEBUG_MODE']:
					print(Pinv)
					print("		PCG: Exiting with err[", abs(nu_prime), "]")
				break
			else:
				if options['DEBUG_MODE']:
					print("		PCG: Iter[", iteration, "] with err[", abs(nu_prime), "]")
			
			beta = nu_prime/nu
			p = r_tilde+p*beta
			nu = nu_prime

		trace = list(map(abs,trace))
		return x, (trace, trace2)

	def compute_preconditioner(self, A, block_size, preconditioner_type):
		if preconditioner_type == "0": # null aka identity
			if(self.overloading):
				return np.identity(A.shape[0])
			else:
				return matrix_(np.identity(A.shape[0]))

		if(self.overloading):
			if preconditioner_type == "J": # Jacobi aka Diagonal
				return matrix_.invert_matrix(matrix_.diag(matrix_.diag(A)))

			elif preconditioner_type == "BJ": # Block-Jacobi
				n_blocks = int(A.shape[0]/block_size)
				Pinv = matrix_(np.zeros(A.shape))
				for k in range(n_blocks):
					rc_k = k*block_size
					rc_kp1 = rc_k+block_size
					Pinv[rc_k:rc_kp1, rc_k:rc_kp1] = matrix_.invert_matrix(A[rc_k:rc_kp1, rc_k:rc_kp1])
				return Pinv

			elif preconditioner_type == "SS": # Symmetric Stair (for blocktridiagonal of blocksize nq+nv)
				n_blocks = int(A.shape[0] / block_size)
				Pinv = matrix_(np.zeros(A.shape))
				# compute stair inverse
				for k in range(n_blocks):
					# compute the diagonal term
					Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size] = \
						matrix_.invert_matrix(A[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size])
					if np.mod(k, 2): # odd block includes off diag terms
						# Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
						Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size] = \
							-(Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size]@ \
									(A[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size]@ \
												Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size]))
					elif k > 0: # compute the off diag term for previous odd block (if it exists)
						# Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
						Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
							-(Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size]@ \
									(A[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size]@ \
												Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size]))
				# make symmetric
				for k in range(n_blocks):
					if np.mod(k, 2): # copy from odd blocks
						# always copy up the left to previous right
						Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
							Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size].transpose()
						# if not last block copy right to next left
						if k < n_blocks - 1:
							Pinv[(k+1)*block_size:(k+2)*block_size, k*block_size:(k+1)*block_size] = \
								Pinv[k*block_size:(k+1)*block_size, (k+1)*block_size:(k+2)*block_size].transpose()
							
				return Pinv
		
		else:
							
			if preconditioner_type == "J": # Jacobi aka Diagonal
				return np.linalg.inv(np.diag(np.diag(A)))

			elif preconditioner_type == "BJ": # Block-Jacobi
				n_blocks = int(A.shape[0] / block_size)
				Pinv = np.zeros(A.shape)
				for k in range(n_blocks):
					rc_k = k*block_size
					rc_kp1 = rc_k + block_size
					Pinv[rc_k:rc_kp1, rc_k:rc_kp1] = np.linalg.inv(A[rc_k:rc_kp1, rc_k:rc_kp1])

				return Pinv

			elif preconditioner_type == "SS": # Symmetric Stair (for blocktridiagonal of blocksize nq+nv)
				n_blocks = int(A.shape[0] / block_size)
				Pinv = np.zeros(A.shape)
				# compute stair inverse
				for k in range(n_blocks):
					# compute the diagonal term
					Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size] = \
						np.linalg.inv(A[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size])
					if np.mod(k, 2): # odd block includes off diag terms
						# Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
						Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size] = \
							-np.matmul(Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size], \
									np.matmul(A[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size], \
												Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size]))
					elif k > 0: # compute the off diag term for previous odd block (if it exists)
						# Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
						Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
							-np.matmul(Pinv[(k-1)*block_size:k*block_size, (k-1)*block_size:k*block_size], \
									np.matmul(A[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size], \
												Pinv[k*block_size:(k+1)*block_size, k*block_size:(k+1)*block_size]))
				# make symmetric
				for k in range(n_blocks):
					if np.mod(k, 2): # copy from odd blocks
						# always copy up the left to previous right
						Pinv[(k-1)*block_size:k*block_size, k*block_size:(k+1)*block_size] = \
							Pinv[k*block_size:(k+1)*block_size, (k-1)*block_size:k*block_size].transpose()
						# if not last block copy right to next left
						if k < n_blocks - 1:
							Pinv[(k+1)*block_size:(k+2)*block_size, k*block_size:(k+1)*block_size] = \
								Pinv[k*block_size:(k+1)*block_size, (k+1)*block_size:(k+2)*block_size].transpose()
							
				return Pinv

	def solve(self):
	    return self.pcg(self.A, self.b, self.Pinv, self.guess, self.options)