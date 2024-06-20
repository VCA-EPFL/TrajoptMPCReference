import importlib
import numpy as np
import copy
import enum
from TrajoptPlant import TrajoptPlant,URDFPlant
from TrajoptCost import TrajoptCost, QuadraticCost
from TrajoptConstraint import TrajoptConstraint, BoxConstraint
PCG = importlib.import_module("GBD-PCG-Python").PCG
np.set_printoptions(precision=4, suppress=True, linewidth = 100)
from overloading import matrix_


class SQPSolverMethods(enum.Enum):
    N = "N"
    S = "S"
    PCG_J = "PCG-J"
    PCG_BJ = "PCG-BJ"
    PCG_SS = "PCG-SS"


class MPCSolverMethods(enum.Enum):
    iLQR = "iLQR"
    QP_N = "QP-N"
    QP_S = "QP-S"
    QP_PCG_J = "QP-PCG-J"
    QP_PCG_BJ = "QP-PCG-BJ"
    QP_PCG_SS = "QP-PCG-SS"

class TrajoptMPCReference:
    
    def __init__(self, plantObj:TrajoptPlant, costObj: TrajoptCost, constraintObj: TrajoptConstraint = None):
        if (not issubclass(type(plantObj),TrajoptPlant) or not issubclass(type(costObj),TrajoptCost)):
            print("Must pass in a TrajoptPlant and TrajoptCost object to TrajoptMPCReference.")
            exit()
        if constraintObj is None:
            constraintObj = TrajoptConstraint()
        elif not issubclass(type(constraintObj),TrajoptConstraint):
            print("If passing in additional constraints must pass in a TrajoptConstraint object to TrajoptMPCReference.")
            exit()
        self.plant = plantObj
        self.cost = costObj
        self.other_constraints = constraintObj

        self.n_inner_iter=0

        self.saved_Pinv=[]
        self.saved_inner_traces=[]
        self.trace=[]
        self.saved_G=[]
        self.saved_invG=[]
        self.saved_c=[]
        self.saved_g=[]
        self.saved_C=[]
        self.saved_tot_cost=[]
        self.saved_J_tot_constraints=[]
        self.saved_jacobian_hard_constraints=[]
        self.saved_jacobian_soft_constraints=[]
        self.saved_Ak=[] # plant integrator output 1 (return gradient)
        self.saved_Bk=[] # plant integrator output 2 (return gradient)
        self.saved_xkp1=[] # plant integrator output 
        self.saved_dxul=[]
        self.saved_S=[]
        self.saved_gamma=[]
        self.saved_l=[]
        self.saved_invG=[]



        self.saved_x=[]
        self.saved_u=[]

        self.exit_soft=0
        self.exit_sqp=0
        self.singular=False




    def update_cost(self, costObj: TrajoptCost):
        assert issubclass(type(costObj),TrajoptCost), "Must pass in a TrajoptCost object to update_cost in TrajoptMPCReference."
        self.cost = costObj

    def update_plant(self, plantObj: TrajoptPlant):
        assert issubclass(type(plantObj),TrajoptPlant), "Must pass in a TrajoptPlant object to update_plant in TrajoptMPCReference."
        self.plant = plantObj

    def update_constraints(self, constraintObj: TrajoptConstraint):
        assert issubclass(type(constraintObj),TrajoptConstraint), "Must pass in a TrajoptConstraint object to update_constraints in TrajoptMPCReference."
        self.other_constraints = constraintObj

    def set_default_options(self, options: dict):
        # LinSys options (mostly for PCG)
        options.setdefault('exit_tolerance_linSys', 1e-6)
        options.setdefault('max_iter_linSys', 100)
        options.setdefault('DEBUG_MODE_linSys', False)
        options.setdefault('RETURN_TRACE_linSys', False)
        options.setdefault('overloading',self.plant.rbdReference.overloading)
        # DDP/SQP options
        options.setdefault('exit_tolerance_SQP_DDP', 1e-6)
        options.setdefault('max_iter_SQP_DDP', 100)
        options.setdefault('DEBUG_MODE_SQP_DDP', False)
        options.setdefault('alpha_factor_SQP_DDP', 0.5)
        options.setdefault('alpha_min_SQP_DDP', 0.005)
        options.setdefault('rho_factor_SQP_DDP', 4)
        options.setdefault('rho_min_SQP_DDP', 1e-3)
        options.setdefault('rho_max_SQP_DDP', 1e3)
        options.setdefault('rho_init_SQP_DDP', 0.001)
        options.setdefault('expected_reduction_min_SQP_DDP', 0.05)
        options.setdefault('expected_reduction_max_SQP_DDP', 3)
        # SQP only options
        options.setdefault('merit_factor_SQP', 1.5)
        # AL and ADMM options
        options.setdefault('exit_tolerance_softConstraints', 1e-6)
        options.setdefault('max_iter_softConstraints', 10)
        options.setdefault('DEBUG_MODE_Soft_Constraints', False)


    def formKKTSystemBlocks(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        if(self.plant.rbdReference.overloading):
        
            total_states_controls = n*(N-1) + nx
            G = matrix_(np.zeros((total_states_controls, total_states_controls)))
            g = matrix_(np.zeros((total_states_controls, 1)))
            total_dynamics_intial_state_constraints = nx*N
            total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
            total_constraints = total_dynamics_intial_state_constraints+total_other_constraints
            C = matrix_(np.zeros((total_constraints, total_states_controls)))
            c = matrix_(np.zeros((total_constraints, 1)))

            constraint_index = 0
            state_control_index = 0
            C[constraint_index:constraint_index + nx, state_control_index:state_control_index + nx] = matrix_(np.eye(nx))
            c[constraint_index:constraint_index + nx, 0] = x[:,0]-xs
            constraint_index += nx
            for k in range(N-1):
                hess=self.cost.hessian(x[:,k], u[:,k], k,iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
                grad=self.cost.gradient(x[:,k], u[:,k], k, iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
                G[state_control_index:state_control_index + n, \
                state_control_index:state_control_index + n] = hess
                g[state_control_index:state_control_index + n, 0] = grad
                if self.other_constraints.total_soft_constraints(timestep = k) > 0:
                    gck = self.other_constraints.jacobian_soft_constraints(x[:,k], u[:,k], k)
                    g[state_control_index:state_control_index + n, :] = g[state_control_index:state_control_index + n, :]+gck
                    G[state_control_index:state_control_index + n, \
                    state_control_index:state_control_index + n] = G[state_control_index:state_control_index + n, state_control_index:state_control_index + n]+matrix_(np.outer(gck,gck))
                    self.saved_J_tot_constraints.append({'value': gck, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

                Ak, Bk = self.plant.integrator(x[:,k], u[:,k], dt, return_gradient = True,iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
                C[constraint_index:constraint_index + nx, \
                state_control_index:state_control_index + n + nx] = matrix_.hstack(-Ak, -Bk, matrix_(np.eye(nx)))
                xkp1 = self.plant.integrator(x[:,k], u[:,k], dt, iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
                c[constraint_index:constraint_index + nx, 0] = x[:,k+1]-xkp1
                constraint_index += nx

                self.saved_Ak.append({'value': Ak,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                self.saved_Bk.append({'value': Bk,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                self.saved_xkp1.append({'value': xkp1, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                
                if total_other_constraints > 0 and self.other_constraints.total_hard_constraints(x, u, k):
                    jac = self.other_constraints.jacobian_hard_constraints(x[:,k], u[:,k], k)
                    val = self.other_constraints.value_hard_constraints(x[:,k], u[:,k], k)
                    self.saved_jacobian_hard_constraints.append({'value': jac, 'iteration': matrix_.iteration,'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                    if val is not None and len(val):
                        num_active_const_k = len(val)
                        C[constraint_index:constraint_index + num_active_const_k, \
                        state_control_index:state_control_index + n] = matrix_.reshape(jac,(num_active_const_k,n))
                        c[constraint_index:constraint_index + num_active_const_k] = matrix_.reshape(val, (num_active_const_k,1))
                        constraint_index += num_active_const_k
                
                state_control_index += n
            hess= self.cost.hessian(x[:,N-1], timestep = N-1, iter_1= matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
            grad= self.cost.gradient(x[:,N-1], timestep = N-1,iter_1= matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
            G[state_control_index:state_control_index + nx, \
            state_control_index:state_control_index + nx] =hess
            g[state_control_index:state_control_index + nx, 0] = grad
           
            if self.other_constraints.total_soft_constraints(timestep = N-1) > 0:
                gcNm1 = self.other_constraints.jacobian_soft_constraints(x[:,N-1], timestep = N-1)
                g[state_control_index:state_control_index + nx, :] = g[state_control_index:state_control_index + nx, :]+gcNm1
                G[state_control_index:state_control_index + nx, \
                state_control_index:state_control_index + nx] = G[state_control_index:state_control_index + nx, state_control_index:state_control_index + nx]+matrix_(np.outer(gcNm1,gcNm1))
                self.saved_jacobian_soft_constraints.append({'value': gcNm1, 'iteration':matrix_.iteration,'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

            if total_other_constraints > 0 and self.other_constraints.total_hard_constraints(x, u, N-1):
                jac = self.other_constraints.jacobian_hard_constraints(x[:,N-1], timestep = N-1)
                val = self.other_constraints.value_hard_constraints(x[:,N-1], timestep = N-1)
                self.saved_jacobian_hard_constraints.append({'value': jac,'iteration': matrix_.iteration,'outer_iteration':self.jacobian_hard_constraints, 'line_search_iteration': matrix_.line_search_iteration})

                if val is not None and len(val):
                    num_active_const_k = len(val)
                    C[constraint_index:constraint_index + num_active_const_k, \
                    state_control_index:state_control_index + nx] = matrix_.reshape(jac, (num_active_const_k,nx))
                    c[constraint_index:constraint_index + num_active_const_k] = matrix_.reshape(val, (num_active_const_k,1))

        else:
                    
            total_states_controls = n*(N-1) + nx
            G = np.zeros((total_states_controls, total_states_controls))
            g = np.zeros((total_states_controls, 1))
            total_dynamics_intial_state_constraints = nx*N
            total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
            total_constraints = total_dynamics_intial_state_constraints + total_other_constraints
            C = np.zeros((total_constraints, total_states_controls))
            c = np.zeros((total_constraints, 1))

            constraint_index = 0
            state_control_index = 0
            C[constraint_index:constraint_index + nx, state_control_index:state_control_index + nx] = np.eye(nx)
            c[constraint_index:constraint_index + nx, 0] = x[:,0]-xs
            constraint_index += nx
            for k in range(N-1):
                G[state_control_index:state_control_index + n, \
                state_control_index:state_control_index + n] = self.cost.hessian(x[:,k], u[:,k], k,iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
                g[state_control_index:state_control_index + n, 0] = self.cost.gradient(x[:,k], u[:,k], k,iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
                if self.other_constraints.total_soft_constraints(timestep = k) > 0:
                    gck = self.other_constraints.jacobian_soft_constraints(x[:,k], u[:,k], k)
                    g[state_control_index:state_control_index + n, :] = g[state_control_index:state_control_index + n, :]+gck
                    G[state_control_index:state_control_index + n, \
                    state_control_index:state_control_index + n] += np.outer(gck,gck)
                    self.saved_J_tot_constraints.append({'value': gck, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration,'line_search_iteration':  matrix_.line_search_iteration})
                
                Ak, Bk = self.plant.integrator(x[:,k], u[:,k], dt, return_gradient = True, iter_1= matrix_.iteration, iter_2=matrix_.soft_constraint_iteration)
                C[constraint_index:constraint_index + nx, \
                state_control_index:state_control_index + n + nx] = np.hstack((-Ak, -Bk, np.eye(nx)))
                xkp1 = self.plant.integrator(x[:,k], u[:,k], dt,iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration)
                c[constraint_index:constraint_index + nx, 0] = x[:,k+1]-xkp1
                constraint_index += nx
                
                self.saved_Ak.append({'value': Ak,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                self.saved_Bk.append({'value':  Bk, 'iteration':matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                self.saved_xkp1.append({'value': xkp1, 'iteration':matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

                if total_other_constraints > 0 and self.other_constraints.total_hard_constraints(x, u, k):
                    jac = self.other_constraints.jacobian_hard_constraints(x[:,k], u[:,k], k)
                    val = self.other_constraints.value_hard_constraints(x[:,k], u[:,k], k)
                    self.saved_jacobian_hard_constraints.append({'value': jac,'iteration': matrix_.iteration,'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

                    if val is not None and len(val):
                        num_active_const_k = len(val)
                        C[constraint_index:constraint_index + num_active_const_k, \
                        state_control_index:state_control_index + n] = np.reshape(jac, (num_active_const_k,n))
                        c[constraint_index:constraint_index + num_active_const_k] = np.reshape(val, (num_active_const_k,1))
                        constraint_index += num_active_const_k
                
                state_control_index += n

            G[state_control_index:state_control_index + nx, \
            state_control_index:state_control_index + nx] = self.cost.hessian(x[:,N-1], timestep = N-1, iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
            g[state_control_index:state_control_index + nx, 0] = self.cost.gradient(x[:,N-1], timestep = N-1, iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
            if self.other_constraints.total_soft_constraints(timestep = N-1) > 0:
                gcNm1 = self.other_constraints.jacobian_soft_constraints(x[:,N-1], timestep = N-1)
                g[state_control_index:state_control_index + nx, :] = g[state_control_index:state_control_index + nx, :]+gcNm1
                G[state_control_index:state_control_index + nx, \
                state_control_index:state_control_index + nx] = G[state_control_index:state_control_index + nx, state_control_index:state_control_index + nx]+np.outer(gcNm1,gcNm1)
                self.saved_jacobian_soft_constraints.append({'value': gcNm1, 'iteration': matrix_.iteration,'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

            if total_other_constraints > 0 and self.other_constraints.total_hard_constraints(x, u, N-1):
                jac = self.other_constraints.jacobian_hard_constraints(x[:,N-1], timestep = N-1)
                val = self.other_constraints.value_hard_constraints(x[:,N-1], timestep = N-1)
                self.saved_jacobian_hard_constraints.append({'value': jac, 'iteration': matrix_.iteration,'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                if val is not None and len(val):
                    num_active_const_k = len(val)
                    C[constraint_index:constraint_index + num_active_const_k, \
                    state_control_index:state_control_index + nx] = np.reshape(jac, (num_active_const_k,nx))
                    c[constraint_index:constraint_index + num_active_const_k] = np.reshape(val, (num_active_const_k,1))
        return G, g, C, c

    def totalHardConstraintViolation(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float, mode = None):
        mode_func = sum
        if mode == "MAX":
            mode_func = max
        # first do initial state and dynamics
        x_err = x[:,0]-xs
        err = list(map(abs,x_err))
        c = mode_func(err)
        for k in range(N-1):
            xkp1 = self.plant.integrator(x[:,k], u[:,k], dt,iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
            x_err = x[:,k+1]-xkp1
            c = c+mode_func(list(map(abs,x_err)))
        # then do all other constraints
        if self.other_constraints.total_hard_constraints(x, u) > 0:
            for k in range(N-1):
                if self.other_constraints.total_hard_constraints(x, u, k):
                    c_err = self.other_constraints.value_hard_constraints(x[:,k], u[:,k], k)
                    c = c+mode_func(list(map(abs,c_err)))
            if self.other_constraints.total_hard_constraints(x, u, N-1):
                c_err = self.other_constraints.value_hard_constraints(x[:,N-1], N-1)
                c = c+mode_func(list(map(abs,c_err)))
        return c

    def totalCost(self, x: np.ndarray, u: np.ndarray, N: int):
        
        J = 0
        for k in range(N-1):
            cost=self.cost.value(x[:,k], u[:,k], k, matrix_.iteration, matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
            J = J+cost
        J = J+self.cost.value(x[:,N-1], timestep = N-1, iter_1=matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)
        # add soft constraints if applicable
        if self.other_constraints.total_soft_constraints() > 0:
            for k in range(N-1):
                J = J+self.other_constraints.value_soft_constraints(x[:,k], u[:,k], k)
            J = J+self.other_constraints.value_soft_constraints(x[:,N-1], timestep = N-1)
        
        self.saved_tot_cost.append({'value': J,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        return J


    def solveKKTSystem(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float, rho: float = 0.0, options = {}):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu
        
        G, g, C, c = self.formKKTSystemBlocks(x, u, xs, N, dt)

        self.saved_G.append({'value':G,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        self.saved_g.append({'value':g,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        self.saved_C.append({'value':C,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        self.saved_c.append({'value':c,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

        total_dynamics_intial_state_constraints = nx*N
        total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
        total_constraints = total_dynamics_intial_state_constraints+total_other_constraints

        if(self.plant.rbdReference.overloading):
            BR = matrix_(np.zeros((total_constraints,total_constraints)))

            if rho != 0:
                G = G+(rho*matrix_(np.eye(G.shape[0])))

            KKT = matrix_.hstack(matrix_.vstack(G, C),matrix_.vstack(C.transpose(), BR))
            kkt = matrix_.vstack(g, c)

            dxul, self.singular= matrix_.linalg_solve(KKT, kkt)
            

        else:
            BR = np.zeros((total_constraints,total_constraints))

            if rho != 0:
                G = G+(rho * np.eye(G.shape[0]))

            KKT = np.hstack((np.vstack((G, C)),np.vstack((C.transpose(), BR))))
            # KKT = np.hstack((np.vstack((G, C)),np.vstack((C.transpose(), BR))))
            kkt = np.vstack((g, c))

            try:
                dxul = np.linalg.solve(KKT, kkt)
            except:
                self.singular= True #Warning singular KKT system -- solving with least squares
                dxul, _, _, _ = np.linalg.lstsq(KKT, kkt, rcond=None)

        return dxul

    def solveKKTSystem_Schur(self, x: np.ndarray, u: np.ndarray, xs: np.ndarray, N: int, dt: float, rho: float = 0.0, use_PCG = False, options = {}):
        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        
        G, g, C, c = self.formKKTSystemBlocks(x, u, xs, N, dt)

        self.saved_G.append({'value': G,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        self.saved_g.append({'value': g,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        self.saved_C.append({'value': C,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        self.saved_c.append({'value': c,'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
        
        
        total_dynamics_intial_state_constraints = nx*N
        total_other_constraints = self.other_constraints.total_hard_constraints(x, u)
        total_constraints = total_dynamics_intial_state_constraints+total_other_constraints

        if(self.plant.rbdReference.overloading):
            
            BR = matrix_(np.zeros((total_constraints,total_constraints)))

            if rho != 0:
                G = G+(rho*matrix_(np.eye(G.shape[0])))
                       
            invG = matrix_.invert_matrix(G)
            self.saved_invG.append({'value': invG, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

            # compute cond of G
            S = BR-(C@(invG@C.transpose()))
            gamma = c-(C@(invG@g))
            self.saved_invG.append({'value': invG, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            self.saved_S.append({'value': S, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            self.saved_gamma.append({'value': gamma, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            if not use_PCG:
                l = matrix_.linalg_solve(S, gamma)
            else:

                pcg = PCG(S, gamma, nx, N, options = options, overloading= True)
                if 'guess' in options.keys():
                    pcg.update_guess(options['guess'])
                l, traces = pcg.solve()


                self.saved_inner_traces.append((traces, matrix_.iteration,matrix_.soft_constraint_iteration, matrix_.line_search_iteration))
                self.saved_Pinv.append({'value': pcg.Pinv, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                self.n_inner_iter= len(traces)
            self.saved_l.append({'value': l, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            

            gCl = g-(C.transpose()@l)
            dxu = invG@gCl

            dxul = matrix_.vstack(dxu,l)
        else:

            BR = np.zeros((total_constraints,total_constraints))

            if rho != 0:
                G += rho * np.eye(G.shape[0])
            
            invG = np.linalg.inv(G)
            S = BR - np.matmul(C, np.matmul(invG, C.transpose()))
            gamma = c - np.matmul(C, np.matmul(invG, g))

            self.saved_invG.append({'value': invG, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            self.saved_S.append({'value': S, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            self.saved_gamma.append({'value': gamma, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

            if not use_PCG:
                try:
                    l = np.linalg.solve(S, gamma)
                except:
                    
                    self.singular=True #Warning singular Schur system -- solving with least squares.")
                    l, _, _, _ = np.linalg.lstsq(S, gamma, rcond=None)
            else:
                pcg = PCG(S, gamma, nx, N, options = options, overloading=False)
                if 'guess' in options.keys():
                    pcg.update_guess(options['guess'])

                l, traces,  = pcg.solve()
                self.saved_inner_traces.append((traces, matrix_.iteration, matrix_.soft_constraint_iteration))
                self.saved_Pinv.append({'value': pcg.Pinv, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                self.n_inner_iter=len(traces)
            
            self.saved_l.append({'value': l, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
            
            gCl = g - np.matmul(C.transpose(), l)
            dxu = np.matmul(invG, gCl)

            dxul = np.vstack((dxu,l))


        return dxul

    def reduce_regularization(self, rho: float, drho: float, options: dict):
        self.set_default_options(options)
        drho = min(drho/options['rho_factor_SQP_DDP'], 1/options['rho_factor_SQP_DDP'])
        rho = max(rho*drho, options['rho_min_SQP_DDP'])
        return rho, drho

    def check_for_exit_or_error(self, error: bool, delta_J: float, iteration: int, rho: float, drho: float, options):
        self.set_default_options(options)
        exit_flag = False
        if error:
            drho = max(drho*options['rho_factor_SQP_DDP'], options['rho_factor_SQP_DDP'])
            rho = max(rho*drho, options['rho_min_SQP_DDP'])
            if rho > options['rho_max_SQP_DDP']:
                self.exit_sqp= 2 # Exiting for max_rho
                exit_flag = True
        elif delta_J < options['exit_tolerance_SQP_DDP']:
            self.exit_sqp= 1 # Exiting for exit_tolerance_SQP_DDP
            exit_flag = True
        
        if iteration == options['max_iter_SQP_DDP'] - 1:
            self.exit_sqp=3 # Exiting for max_iter
            exit_flag = True
        else:
            iteration += 1
        return exit_flag, iteration, rho, drho

    def check_and_update_soft_constraints(self, x: np.ndarray, u: np.ndarray, iteration: int, options):
        exit_flag = False
        # check for exit for constraint convergence
        max_c = self.other_constraints.max_soft_constraint_value(x,u)
        if max_c < options['exit_tolerance_softConstraints']:

            # if options['DEBUG_MODE_Soft_Constraints']:
            self.exit_soft = 1 # OUTER LOOP: Exiting for Soft Constraint Convergence
            exit_flag = True
        # check for exit for iterations
        if iteration == options['max_iter_softConstraints'] - 1:

            # if options['DEBUG_MODE_Soft_Constraints']:
            self.exit_soft = 2 # OUTER LOOP: Exiting for Soft Constraint Max Iters
            exit_flag = True
        else:
            iteration += 1
        # if we are not exiting update soft constraint constants
        if not exit_flag:
            all_mu_over_limit_flag = self.other_constraints.update_soft_constraint_constants(x,u)
            # check if we need to exit for mu over the limit
            if all_mu_over_limit_flag:
                # if options['DEBUG_MODE_Soft_Constraints']:
                self.exit_soft = 3 # OUTER LOOP: Exiting for Mu over limit for all soft constraints
                exit_flag = True
        return exit_flag, iteration

    def SQP(self, x: np.ndarray, u: np.ndarray, N: int, dt: float, LINEAR_SYSTEM_SOLVER_METHOD: SQPSolverMethods = SQPSolverMethods.N, options = {}):
        self.set_default_options(options)
        options_linSys = {'DEBUG_MODE': options['DEBUG_MODE_linSys']}

        USING_PCG = LINEAR_SYSTEM_SOLVER_METHOD in [SQPSolverMethods.PCG_J, SQPSolverMethods.PCG_BJ, SQPSolverMethods.PCG_SS]
        if USING_PCG:
            options_linSys['exit_tolerance'] = options['exit_tolerance_linSys']
            options_linSys['max_iter'] = options['max_iter_linSys']
            options_linSys['RETURN_TRACE'] = options['RETURN_TRACE_linSys']
            options_linSys['preconditioner_type'] = LINEAR_SYSTEM_SOLVER_METHOD.value[4:]

        nq = self.plant.get_num_pos()
        nv = self.plant.get_num_vel()
        nu = self.plant.get_num_cntrl()
        nx = nq + nv
        n = nx + nu

        xs = copy.deepcopy(x[:,0])

        # Start the main loops (soft constraint outer loop)
        matrix_.soft_constraint_iteration = 0
        while 1:
            # print("Inside Soft constraints loop")

            # Initialize the QP solve
            J = 0
            c = 0
            rho = options['rho_init_SQP_DDP']
            drho = 1

            # Compute initial cost and constraint violation
            J = self.totalCost(x, u, N)
            c = self.totalHardConstraintViolation(x, u, xs, N, dt)

            # L1 merit function with balanced J and c
            mu = J/c if c != 0 else 10
            mu = 10

            merit = J + mu*c
           

            if options['DEBUG_MODE_SQP_DDP']:
                print("Initial Cost, Constraint Violation, Merit Function: ", J, c, merit)
            
            inner_iters = 0
            self.trace = [{
                'outer_iteration': matrix_.soft_constraint_iteration,
                'iteration': 0,
                'line_search_iteration': 0,
                'alpha': 1,
                'rho': rho,
                'J': J,
                'c': c,
                'merit': merit,
                'D': None,
                'reduction_ratio': None,
                'inner_iters': self.n_inner_iter,
                'singular': False,
                'succeeded_line_search': False
            }]

            # Start the main loop (SQP main loop)
            matrix_.iteration = 0
            while 1:

                #
                # Solve QP to get step direction
                #
                # print("Inside SQP loop")

                # print("Defining QP problem : KKT")
                # print("Solving QP problem")
                self.n_inner_iter=0
                if LINEAR_SYSTEM_SOLVER_METHOD == SQPSolverMethods.N: # standard backslash
                    dxul = self.solveKKTSystem(x, u, xs, N, dt, rho, options_linSys)
                elif LINEAR_SYSTEM_SOLVER_METHOD == SQPSolverMethods.S: # schur complement backslash
                    dxul = self.solveKKTSystem_Schur(x, u, xs, N, dt, rho, False, options_linSys)
                elif USING_PCG: # PCG
                    dxul = self.solveKKTSystem_Schur(x, u, xs, N, dt, rho, True, options_linSys)
                
                else:
                    print("Invalid QP Solver options are:\n", \
                          "N      : Standard Backslash\n", \
                          "S      : Schur Complement Backslash\n", \
                          "PCG-X  : PCG with Preconditioner X (see PCG for valid preconditioners)\n")
                    print("If calling from SQP the solver must be called QP-X where X is a solver option above.")
                    exit()

                self.saved_dxul.append({'value': dxul, 'iteration': matrix_.iteration, 'outer_iteration': matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})

                
        

                #
                # Do line search and accept iterate or regularize the problem
                #
                alpha = 1
                error = False
                matrix_.line_search_iteration =0
                while 1:

                    # print("     Start of line seach loop, iteration:", matrix_.iteration )
                    #
                    # Apply the update
                    #
                    # print("     Update x and u trajectories with the (N size) search direction found in QP solve ")

                    x_new = copy.deepcopy(x)
                    u_new = copy.deepcopy(u)
                    for k in range(N):
                        x_new[:,k] = x_new[:,k]-alpha*dxul[n*k : n*k+nx, 0]
                        if k < N-1:
                            u_new[:,k] = u_new[:,k]-alpha*dxul[n*k+nx : n*(k+1), 0]

                    # print("     Compute the cost, constraint violation, and directional derivative")
                    
                    #
                    # Compute the cost, constraint violation, and directional derivative
                    #
                    J_new = self.totalCost(x_new, u_new, N)
                    c_new = self.totalHardConstraintViolation(x_new, u_new, xs, N, dt)
                    
                    #
                    # Directional derivative = grad_J*p - mu|c|
                    #
                    D = 0 
                    for k in range(N-1):
                        D += float(self.cost.gradient(x_new[:,k], u_new[:,k], k,matrix_.iteration, matrix_.soft_constraint_iteration, matrix_.line_search_iteration)@dxul[n*k : n*(k+1), 0])
                        
                        # Add soft constraints if applicable
                        if self.other_constraints.total_soft_constraints(timestep = k) > 0:
                            D += float(self.other_constraints.jacobian_soft_constraints(x_new[:,k], u_new[:,k], k)[:,0].dot(dxul[n*k : n*(k+1), 0]))

                    # D += np.dot(self.cost.gradient(x_new[:,N-1], timestep = N-1), dxul[n*(N-1) : n*(N-1)+nx, 0])
                    D += float(self.cost.gradient(x_new[:,N-1], timestep = N-1, iter_1= matrix_.iteration, iter_2=matrix_.soft_constraint_iteration, iter_3=matrix_.line_search_iteration)@dxul[n*(N-1) : n*(N-1)+nx, 0])
                    # Add soft constraints if applicable
                    if self.other_constraints.total_soft_constraints(timestep = N-1) > 0:
                        #D += np.dot(self.other_constraints.jacobian_soft_constraints(x_new[:,N-1], timestep = N-1)[:,0], dxul[n*(N-1) : n*(N-1)+nx, 0])
                        D += float(self.other_constraints.jacobian_soft_constraints(x_new[:,N-1], timestep = N-1)[:,0].dot(dxul[n*(N-1) : n*(N-1)+nx, 0]))

                   
                    # print("     Compute new merit function")
                    #
                    # Compute totals for line search test
                    #
                    merit_new = J_new+mu*c_new
                    delta_J = J - J_new
                    delta_c = c - c_new
                    delta_merit = merit -  merit_new
                    expected_reduction = alpha * (D - mu * c_new)
                    reduction_ratio = delta_merit/expected_reduction
                    
                    #
                    # If succeeded accept new trajectory according to Nocedal and Wright 18.3
                    #
                    if (delta_merit >= 0 and reduction_ratio >= options['expected_reduction_min_SQP_DDP'] and \
                                             reduction_ratio <= options['expected_reduction_max_SQP_DDP']):
                        
                        # print("     Merit function sufficiently reduced")
                        # print("     Applying updated x trajectory to the actual state + updating actual merit, J and c")
                        x = x_new
                        u = u_new
                        J = J_new
                        c = c_new
                        self.saved_x.append({'value': x, 'iteration': matrix_.iteration, 'outer_iteration': matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                        self.saved_u.append({'value': u, 'iteration': matrix_.iteration, 'outer_iteration':matrix_.soft_constraint_iteration, 'line_search_iteration': matrix_.line_search_iteration})
                        merit = merit_new
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("Iter[", matrix_.iteration, "] Cost[", J_new, "], Constraint Violation[", c_new, "], mu [", mu, "], Merit Function[", merit_new, "] and Reduction Ratio[", reduction_ratio, "] and rho [", rho, "]")
                        # print("     Reduce rho")
                        # update regularization
                        rho, drho = self.reduce_regularization(rho, drho, options)
                        # Check feasability gain vs. optimality gain and adjust mu accordingly
                        # if delta_J/J > delta_c/c:
                        #     mu = min(mu * merit_factor_SQP, 1000)
                        # else:
                        #     mu = max(mu / merit_factor_SQP, 1)
                        # merit = J + mu * c
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("      updated merit: ", merit, " <<< delta J vs c: ", delta_J, " ", delta_c)
                            
                        self.trace.append({
                            'outer_iteration': matrix_.soft_constraint_iteration,
                            'iteration': matrix_.iteration,
                            'line_search_iteration': matrix_.line_search_iteration, #int(-np.log(alpha) / np.log(2)) + 1,
                            'alpha': alpha,
                            'rho': rho,
                            'J': J,
                            'c': c,
                            'merit': merit,
                            'D': D,
                            'reduction_ratio': reduction_ratio,
                            'inner_iters': self.n_inner_iter, #inner_iters,
                            'singular': self.singular,
                            'succeeded_line_search': True
                        })
                        # end line search
                        break
                    
                    #
                    # If failed iterate decrease alpha and try line search again
                    #
                    elif alpha > options['alpha_min_SQP_DDP']:
                        # print("     Failed to reduce Merit function sufficiently, decrease alpha")

                        if options['DEBUG_MODE_SQP_DDP']:
                            print("Alpha[", alpha, "] Rejected with Cost[", J_new, "], Constraint Violation[", c_new, "], mu [", mu, "], Merit Function[", merit_new, "] and Reduction Ratio[", reduction_ratio, "]")
                        alpha *= options['alpha_factor_SQP_DDP']
                        matrix_.line_search_iteration +=1
                    #
                    # If failed the whole line search report the error
                    #
                    else:
                        print("Line search failed")

                        error = True
                        if options['DEBUG_MODE_SQP_DDP']:
                            print("Line search failed")

                        self.trace.append({
                            'outer_iteration': matrix_.soft_constraint_iteration,
                            'iteration': matrix_.iteration,
                            'line_search_iteration': matrix_.line_search_iteration, #int(-np.log(alpha) / np.log(2)) + 1,
                            'alpha': alpha,
                            'rho': rho,
                            'J': J,
                            'c': c,
                            'merit': merit,
                            'D': D,
                            'reduction_ratio': reduction_ratio,
                            'inner_iters': self.n_inner_iter,
                            'singular': self.singular,
                            'succeeded_line_search': False
                        })
                        break
                #
                # Check for exit (or error) and adjust accordingly
                #
                exit_flag, matrix_.iteration, rho, drho = self.check_for_exit_or_error(error, delta_J, matrix_.iteration, rho, drho, options)
                if exit_flag:
                    break

            #
            # Outer loop updates of soft constraint hyperparameters (where appropriate)
            #
            exit_flag, matrix_.soft_constraint_iteration = self.check_and_update_soft_constraints(x, u, matrix_.soft_constraint_iteration, options)
            if exit_flag:
                break


        return x, u, self.exit_sqp, self.exit_soft, matrix_.soft_constraint_iteration, matrix_.iteration

   