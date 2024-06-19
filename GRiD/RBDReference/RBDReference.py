import numpy as np
import copy
import sympy as sp
np.set_printoptions(precision=4, suppress=True, linewidth = 100)
from overloading import matrix_

class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj # instance of Robot Object class created by URDFparser``
        self.overloading = False
        self.saved_minv=[]

    def cross_operator(self, v):
        # for any vector v, computes the operator v x 
        # vec x = [wx   0]
        #         [vox wx]
        #(crm in spatial_v2_extended)
        if(self.overloading):
            v_cross = matrix_([0, -v[2], v[1], 0, 0, 0,
                                v[2], 0, -v[0], 0, 0, 0,
                                -v[1], v[0], 0, 0, 0, 0,
                                0, -v[5], v[4], 0, -v[2], v[1], 
                                v[5], 0, -v[3], v[2], 0, -v[0],
                                -v[4], v[3], 0, -v[1], v[0], 0]
                            ).reshape((6,6))
        else:
            v_cross = np.array([0, -v[2], v[1], 0, 0, 0,
                                v[2], 0, -v[0], 0, 0, 0,
                                -v[1], v[0], 0, 0, 0, 0,
                                0, -v[5], v[4], 0, -v[2], v[1], 
                                v[5], 0, -v[3], v[2], 0, -v[0],
                                -v[4], v[3], 0, -v[1], v[0], 0]
                            ).reshape(6,6)
        return(v_cross)
    
    def dual_cross_operator(self, v):

        #(crf in in spatial_v2_extended)
        return(-1 * self.cross_operator(v).T)
    

    def icrf(self, v):
        
        #helper function defined in spatial_v2_extended library, called by idsva()
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        if(self.overloading):
            return -matrix_(res)
        else:
            return -np.asmatrix(res)



    def mxS(self, S, vec, alpha=1.0):

        # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
        # derivative of spatial motion vector = v x m
        # return(alpha * np.dot(self.cross_operator(vec), S))  
        return( alpha*self.cross_operator(vec)@S )     
    
    def fxv_simple(self, m, f):

        # force spatial vector cross product. 
        # return(np.dot(self.dual_cross_operator(m), f))
        return(self.dual_cross_operator(m).dot(f))
        

    def fxv(self, fxVec, timesVec):

        # Fx(fxVec)*timesVec
        #   0  -v(2)  v(1)    0  -v(5)  v(4)
        # v(2)    0  -v(0)  v(5)    0  -v(3)
        #-v(1)  v(0)    0  -v(4)  v(3)    0
        #   0     0     0     0  -v(2)  v(1)
        #   0     0     0   v(2)    0  -v(0)
        #   0     0     0  -v(1)  v(0)    0
        if(self.overloading):
            result = matrix_(np.zeros((6)))
        else:
            result = np.zeros((6))

        result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2] - fxVec[5] * timesVec[4] + fxVec[4] * timesVec[5]
        result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2] + fxVec[5] * timesVec[3] - fxVec[3] * timesVec[5]
        result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1] - fxVec[4] * timesVec[3] + fxVec[3] * timesVec[4]
        result[3] =                                                     -fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
        result[4] =                                                      fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
        result[5] =                                                     -fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
        return result

    def fxS(self, S, vec, alpha = 1.0):

        # force spatial cross product with motion subspace 
        return -self.mxS(S, vec, alpha)

    def vxIv(self, vec, Imat):

        # necessary component in differentiating Iv (product rule).
        # We express I_dot x v as v x (Iv) (see Featherstone 2.14)
        # our core equation of motion is f = d/dt (Iv) = Ia + vx* Iv
        if(self.overloading):
            temp = Imat@vec
            vecXIvec = matrix_(np.zeros((6)))
        else:
            temp = np.matmul(Imat,vec)
            vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec

    """
    End Effector Posiitons

    offests is an array of np matricies of the form (offset_x, offset_y, offset_z, 1)
    """
    def end_effector_positions(self, q, offsets = [np.matrix([[0,1,0,1]])]):
        if(self.overloading):
            eePos_arr = []
            for jid in self.robot.get_leaf_nodes():
                jidChain = sorted(self.robot.get_ancestors_by_id(jid))
                jidChain.append(jid)
                Xmat_hom = matrix_(np.eye(4))
                for ind in jidChain:
                    currX = matrix_(self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind]))
                    Xmat_hom = Xmat_hom@currX
                eePos_xyz1 = Xmat_hom*offsets[0].transpose()
                eePos_arr.append(eePos_xyz1[:2,:])
            return eePos_arr[0]
        else:
            eePos_arr = []
            for jid in self.robot.get_leaf_nodes():
                jidChain = sorted(self.robot.get_ancestors_by_id(jid))
                jidChain.append(jid)
                Xmat_hom = np.eye(4)
                for ind in jidChain:
                    currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                    Xmat_hom = np.matmul(Xmat_hom,currX)
                
                eePos_xyz1 = Xmat_hom * offsets[0].transpose()
                eePos_arr.append(eePos_xyz1[:2,:])
            return eePos_arr[0]

    """
    End Effectors Position Gradients
    """
    def equals_or_hstack(self, obj, col):
        if obj is None:
            obj = col

        else:
            if(self.overloading):
                obj = matrix_.hstack(obj,col)
            else:
                obj = np.hstack((obj,col))

        return obj
    

    #NOT USED
    def symbolic_jacobian(self,offsets = [np.matrix([[0,1,0,1]])]):
        n = self.robot.get_num_joints()
        q_symbols = [sp.Symbol('q{}'.format(i)) for i in range(1, n+1)]

        deePos_arr = []
        for jid in self.robot.get_leaf_nodes():
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)

            # then compute the gradients
            deePos = None
            for dind in range(n):


                if dind not in jidChain:
                    deePos_col = np.zeros((6,1))
                    deePos = self.equals_or_hstack(deePos,deePos_col)
                
                else:
                    # first chain up the transforms
                    Xmat_hom = np.eye(4)
                    for ind in jidChain:
                        if ind == dind: # use derivative
                            currX = self.robot.get_dXmat_hom_by_id(ind)
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_by_id(ind)
                        symbols = currX.free_symbols
                        theta = symbols.pop()
                        currX=currX.subs(theta,q_symbols[ind])
                        Xmat_hom = Xmat_hom*currX

                    # xyz position is easy
                    deePos_xyz1 = Xmat_hom * offsets[0].transpose()

                    deePos_col = deePos_xyz1[:3,:]
                    deePos = self.equals_or_hstack(deePos,deePos_col)

            deePos_arr.append(deePos)
        J=deePos_arr[0][:2,:2]
        return J
        
    #NOT USED
    def jacobian_grad_func(self,offsets = [np.matrix([[0,1,0,1]])]):
        n = self.robot.get_num_joints()
        q_symbols = [sp.Symbol('q{}'.format(i)) for i in range(1, n+1)]
        J=self.symbolic_jacobian(offsets)
        #need symbolic expression to diff
        dJdq = [[sp.diff(J[i, j], q) for q in q_symbols] for i in range(J.shape[0]) for j in range(J.shape[1])]
        return sp.lambdify(q_symbols,dJdq, "numpy")

    # Commented part => product of Xmat and ddXmat
    # dJdq has same elements as J => less compute
    def dJdq(self,q ,offsets = [np.matrix([[0,1,0,1]])]):
        # n = self.robot.get_num_joints()
        # dJdq=np.zeros((2*n,n))
        # jacobian_grad = []
        # for jid in self.robot.get_leaf_nodes():
        #     jidChain = sorted(self.robot.get_ancestors_by_id(jid))
        #     jidChain.append(jid)
        #     deePos = None
        #     for dind in range(n):
        #         if dind not in jidChain:
        #             deePos_col = np.zeros((6,1))
        #             deePos = self.equals_or_hstack(deePos,deePos_col)
        #         else:
        #             Xmat_hom = np.eye(4)
        #             for ind in jidChain:
        #                 if ind == dind: # use second derivative
        #                     currX = self.robot.get_ddXmat_hom_Func_by_id(ind)(q[ind])
        #                 else: # use normal transform
        #                     currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
        #                 Xmat_hom = np.matmul(Xmat_hom,currX)
        #             deePos_xyz1 = Xmat_hom * offsets[0].transpose()
        #             deePos_col = deePos_xyz1#[:3,:]
        #             deePos = self.equals_or_hstack(deePos,deePos_col)
        #     jacobian_grad.append(deePos)

        # dJdq[0,:]= jacobian_grad[0][0,:n]
        # dJdq[1,:]= [dJdq[0,1]]*n
        # dJdq[2,:]= jacobian_grad[0][1,:2]
        # dJdq[3,:]= [dJdq[2,1]]*n
        # return dJdq

        # J_grad[0,:]= jacobian_grad[0][0,:n]
        # J_grad[1,:]= [J_grad[0,1],J_grad[0,1]]
        # J_grad[2,:]= jacobian_grad[0][1,:2]
        # J_grad[3,:]= [J_grad[2,1],J_grad[2,1]]
        # return J_grad

        n = self.robot.get_num_joints()
        if(self.overloading):
            dJdq=matrix_(np.zeros((2*n,n)))
        else:
            dJdq=np.zeros((2*n,n))
        J=self.Jacobian(q,offsets)
        dJdq[0,:] = -J[1,:]
        dJdq[1,:] = [-J[1,1],-J[1,1]]
        dJdq[2,:] = -J[0,:]
        dJdq[3,:] = [J[0,1],J[0,1]]
        return dJdq




    # Commented part => product of Xmat and ddXmat
    # dJdq has same elements as J => less compute
    def d2Jdq2(self,q ,offsets = [np.matrix([[0,1,0,1]])]):
        # n = self.robot.get_num_joints()
        # ddJdq=np.zeros((2*n,n))
        # J_list = []
        # for jid in self.robot.get_leaf_nodes():
        #     jidChain = sorted(self.robot.get_ancestors_by_id(jid))
        #     jidChain.append(jid)
        #     deePos = None
        #     for dind in range(n):
        #         if dind not in jidChain:
        #             deePos_col = np.zeros((6,1))
        #             deePos = self.equals_or_hstack(deePos,deePos_col)
        #         else:
        #             Xmat_hom = np.eye(4)
        #             for ind in jidChain:
                    
        #                 if ind == dind: # use second derivative
        #                     currX = self.robot.get_dddXmat_hom_Func_by_id(ind)(q[ind])
        #                 else: # use normal transform
        #                     currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
        #                 Xmat_hom = np.matmul(Xmat_hom,currX)
        #             deePos_xyz1 = Xmat_hom * offsets[0].transpose()
        #             deePos_col = deePos_xyz1[:3,:]
        #             deePos = self.equals_or_hstack(deePos,deePos_col)

        #     J_list.append(deePos)
        # ddJdq[0,:]= J_list[0][0,:n]
        # ddJdq[1,:]= [ddJdq[0,1]]*n
        # ddJdq[2,:]= J_list[0][1,:n]
        # ddJdq[3,:]= [ddJdq[2,1]]*n


        n = self.robot.get_num_joints()
        if(self.overloading):
            ddJdq=matrix_(np.zeros((2*n,n)))
        else:
            ddJdq=np.zeros((2*n,n))
        J=self.Jacobian(q,offsets) # compute J twice (in dJdq and ddJdq)
        ddJdq[0,:] = -J[0,:]
        ddJdq[1,:] = [-J[0,1],-J[0,1]]
        ddJdq[2,:] = -J[1,:]
        ddJdq[3,:] = [-J[1,1],-J[1,1]]
        return ddJdq

    #d(x,y,vx,vy)/d(q,q_d)
    def jacobian_tot_state(self,q,qd,offsets = [np.matrix([[0,1,0,1]])]):
        if(self.overloading):
            n = self.robot.get_num_joints()
            J1=self.Jacobian(q,offsets)
            dJdq=self.dJdq(q,offsets)
            J2=( dJdq@qd ).reshape((n,n))
            J_top = matrix_.hstack(J1, np.zeros_like(J1))
            J_bottom = matrix_.hstack(J2, J1)
            J = matrix_.vstack(J_top, J_bottom)
            return J
        else:
            n = self.robot.get_num_joints()
            J1=self.Jacobian(q,offsets)
            dJdq=self.dJdq(q,offsets)
            J2=(dJdq@qd).reshape(n,n)
            J_top = np.hstack((J1, np.zeros_like(J1)))
            J_bottom = np.hstack((J2, J1))
            J = np.vstack((J_top, J_bottom))
            return J
    
    #d(x,y)/dq
    def Jacobian(self, q, offsets = [np.matrix([[0,1,0,1]])]):
        if(self.overloading):
            n = self.robot.get_num_joints()
            jacobian = []
            for jid in self.robot.get_leaf_nodes():
                jidChain = sorted(self.robot.get_ancestors_by_id(jid))
                jidChain.append(jid)
                deePos = None
                for dind in range(n):
                    if dind not in jidChain:
                        deePos_col = matrix_(np.zeros((6,1)))
                        deePos = self.equals_or_hstack(deePos,deePos_col)
                    else:
                        Xmat_hom = matrix_(np.eye(4))
                        for ind in jidChain:
                            if ind == dind: # use derivative
                                currX = matrix_(self.robot.get_dXmat_hom_Func_by_id(ind)(q[ind]))
                            else: # use normal transform
                                currX = matrix_(self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind]))
                            Xmat_hom = Xmat_hom@currX
                        deePos_xyz1 = Xmat_hom*offsets[0].transpose()
                        deePos_col = deePos_xyz1[:3,:]
                        deePos = self.equals_or_hstack(deePos,deePos_col)
                jacobian.append(deePos)
            return jacobian[0][:n,:n]
        else:
            n = self.robot.get_num_joints()
            jacobian = []
            for jid in self.robot.get_leaf_nodes():
                jidChain = sorted(self.robot.get_ancestors_by_id(jid))
                jidChain.append(jid)
                deePos = None
                for dind in range(n):
                    if dind not in jidChain:
                        deePos_col = np.zeros((6,1))
                        deePos = self.equals_or_hstack(deePos,deePos_col)
                    else:
                        Xmat_hom = np.eye(4)
                        for ind in jidChain:
                            if ind == dind: # use derivative
                                currX = self.robot.get_dXmat_hom_Func_by_id(ind)(q[ind])
                            else: # use normal transform
                                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                            Xmat_hom = np.matmul(Xmat_hom,currX)
                        deePos_xyz1 = Xmat_hom * offsets[0].transpose()
                        deePos_col = deePos_xyz1[:3,:]
                        deePos = self.equals_or_hstack(deePos,deePos_col)
                jacobian.append(deePos)
            return jacobian[0][:n,:n]

    """
    Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

    RNEA divided into 3 parts: 
        1) calculate the velocity and acceleration of each body in the tree
        2) Calculate the forces necessary to produce these accelertions
        3) Calculate the forces transmitted across the joints from the forces acting on the bodies
    """

    
    def rnea_fpass(self, q, qd, qdd = None, GRAVITY = -9.81):

        """
        Forward Pass for RNEA algorithm. Computes the velocity and acceleration of each body in the tree necessary to produce a certain trajectory
        
        OUTPUT: 
        v : input qd is specifying value within configuration space with assumption of one degree of freedom. 
        Output velocity is in general body coordinates and specifies motion in full 6 degrees of freedom
        """
        assert len(q) == len(qd), "Invalid Trajectories"
        # not sure should equal num links or num joints. 
        assert len(q) == self.robot.get_num_joints(), "Invalid Trajectory, must specify coordinate for every body" 
        if(self.overloading):
            # allocate memory
            n = len(q)
            v = matrix_(np.zeros((6,n)))
            a = matrix_(np.zeros((6,n)))
            f = matrix_(np.zeros((6,n)))

            gravity_vec = matrix_(np.zeros((6)))# model gravity as a fictitious base acceleration. 
            # all forces subsequently offset by gravity. 
            gravity_vec[5] = -GRAVITY # a_base is gravity vec, linear in z direction

            # forward pass
            # vi = vparent + Si * qd_i
            # differentiate for ai = aparent + Si * qddi + Sdi * qdi
            
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind])) # the coordinate transform that brings into base reference frame
                S = matrix_(self.robot.get_S_by_id(ind)) # Subspace matrix
                # compute v and a
                if parent_ind == -1: # parent is base
                    # v_base is zero so v[:,ind] remains 0
                    a[:,ind] = Xmat@gravity_vec
                else:
                    v[:,ind] = Xmat@v[:,parent_ind] # velocity of parent in base coordinates. 
                    a[:,ind] = Xmat@a[:,parent_ind]
                v[:,ind] = v[:,ind]+(S*qd[ind]) # S turns config space into actual velocity

                a[:,ind] = a[:,ind]+self.mxS(S,v[:,ind],qd[ind])
                if qdd is not None:
                    a[:,ind] = a[:,ind]+(S*qdd[ind])

                # compute f
                Imat = matrix_(self.robot.get_Imat_by_id(ind))
                f[:,ind] = Imat@a[:,ind]+matrix_(self.vxIv(v[:,ind],Imat ))
                newv=matrix_([0.,0.,7.,1.])
        else:
            # allocate memory
            n = len(q)
            v = np.zeros((6,n))
            a = np.zeros((6,n))
            f = np.zeros((6,n))

            gravity_vec = np.zeros((6)) # model gravity as a fictitious base acceleration. 
            # all forces subsequently offset by gravity. 
            gravity_vec[5] = -GRAVITY # a_base is gravity vec, linear in z direction

            # forward pass
            # vi = vparent + Si * qd_i
            # differentiate for ai = aparent + Si * qddi + Sdi * qdi
            
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind]) # the coordinate transform that brings into base reference frame
                S = self.robot.get_S_by_id(ind) # Subspace matrix
                # compute v and a
                if parent_ind == -1: # parent is base
                    # v_base is zero so v[:,ind] remains 0
                    a[:,ind] = np.matmul(Xmat,gravity_vec)
                else:
                    v[:,ind] = np.matmul(Xmat,v[:,parent_ind]) # velocity of parent in base coordinates. 
                    a[:,ind] = np.matmul(Xmat,a[:,parent_ind])
                v[:,ind] += S*qd[ind] # S turns config space into actual velocity

                a[:,ind] += self.mxS(S,v[:,ind],qd[ind])
                if qdd is not None:
                    a[:,ind] += S*qdd[ind]

                # compute f
                Imat = self.robot.get_Imat_by_id(ind)
                f[:,ind] = np.matmul(Imat,a[:,ind]) + self.vxIv(v[:,ind],Imat)
                newv=np.array([0.,0.,7.,1.])

        return (v,a,f)

    def rnea_bpass(self, q, qd, f, USE_VELOCITY_DAMPING = False):

        if(self.overloading):
            # allocate memory
            n = len(q) # assuming len(q) = len(qd)
            c = matrix_(np.zeros(n))
            for ind in range(n-1,-1,-1):
                S = matrix_(self.robot.get_S_by_id(ind))
                # compute c
                c[ind] = S.T@f[:,ind]
                # update f if applicable
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                    temp = Xmat.T@f[:,ind]
                    f[:,parent_ind] = f[:,parent_ind]+temp.flatten()

            # add velocity damping (defaults to 0)
            if USE_VELOCITY_DAMPING:
                for k in range(n):
                    c[k] = c[k]+self.robot.get_damping_by_id(k) * qd[k]
        else:
            # allocate memory
            n = len(q) # assuming len(q) = len(qd)
            c = np.zeros(n)
            
            # backward pass
            # seek to calculate force transmitted from body i across joint i (fi) from the outside in.
            # fi = fi^B (net force) - fi^x (external forces, assumed to be known) - sum{f^j (all forces from children)}. 
            # Start with outermost as set of children is empty and go backwards to base.
            for ind in range(n-1,-1,-1):
                S = self.robot.get_S_by_id(ind)
                # compute c
                c[ind] = np.matmul(np.transpose(S),f[:,ind])
                # update f if applicable
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                    temp = np.matmul(np.transpose(Xmat),f[:,ind])
                    f[:,parent_ind] = f[:,parent_ind] + temp.flatten()

            # add velocity damping (defaults to 0)
            if USE_VELOCITY_DAMPING:
                for k in range(n):
                    c[k] += self.robot.get_damping_by_id(k) * qd[k]

        return (c,f)

    def rnea(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
        """
        Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

        RNEA divided into 3 parts: 
            1) calculate the velocity and acceleration of each body in the tree
            2) Calculate the forces necessary to produce these accelertions
            3) Calculate the forces transmitted across the joints from the forces acting on the bodies
            
        INPUT:
        q, qd, qdd: position, velocity, acceleration. Nx1 arrays where N is the number of bodies
        GRAVITY - gravitational field of the body; default is earth surface gravity, 9.81
        USE_VELOCITY_DAMPING: flag for whether velocity is damped, representing ___
        
        OUTPUTS: 
        c: Coriolis terms and other forces potentially be applied to the system. 
        v: velocity of each joint in world base coordinates rather than motion subspace
        a: acceleration of each joint in world base coordinates rather than motion subspace
        f: forces that joints must apply to produce trajectory
        """
        # forward pass
        (v,a,f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
        # backward pass
        (c,f) = self.rnea_bpass(q, qd, f, USE_VELOCITY_DAMPING)

        return (c,v,a,f)

    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):

        if(self.overloading):
            # allocate memory
            n = len(qd)
            dv_dq = matrix_(np.zeros((6,n,n)))
            da_dq = matrix_(np.zeros((6,n,n)))
            df_dq = matrix_(np.zeros((6,n,n)))

            gravity_vec = matrix_(np.zeros((6)))
            gravity_vec[5] = -GRAVITY # a_base is gravity vec

            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                S = matrix_(self.robot.get_S_by_id(ind))
                # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
                if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                    dv_dq[:,:,ind] = Xmat@dv_dq[:,:,parent_ind]
                    dv_dq[:,ind,ind] = dv_dq[:,ind,ind]+self.mxS(S,Xmat@v[:,parent_ind])
                # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
                if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                    da_dq[:,:,ind] = Xmat@da_dq[:,:,parent_ind]
                for c in range(n):
                    da_dq[:,c,ind] = da_dq[:,c,ind]+self.mxS(S,dv_dq[:,c,ind],qd[ind])
                if parent_ind != -1: # note that a_base is just gravity
                    da_dq[:,ind,ind] = da_dq[:,ind,ind]+self.mxS(S,Xmat@a[:,parent_ind])
                else:
                    da_dq[:,ind,ind] = da_dq[:,ind,ind]+self.mxS(S,Xmat@gravity_vec)
                # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
                Imat = matrix_(self.robot.get_Imat_by_id(ind))
                df_dq[:,:,ind] = Imat@da_dq[:,:,ind]
                Iv = Imat@v[:,ind]
                for c in range(n):
                    df_dq[:,c,ind] = df_dq[:,c,ind]+self.fxv(dv_dq[:,c,ind],Iv)
                    df_dq[:,c,ind] = df_dq[:,c,ind]+self.fxv(v[:,ind],Imat@dv_dq[:,c,ind])
        else:
                # allocate memory
            n = len(qd)
            dv_dq = np.zeros((6,n,n))
            da_dq = np.zeros((6,n,n))
            df_dq = np.zeros((6,n,n))

            gravity_vec = np.zeros((6))
            gravity_vec[5] = -GRAVITY # a_base is gravity vec

            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)
                # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
                if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                    dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                    dv_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,v[:,parent_ind]))
                # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
                if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                    da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
                for c in range(n):
                    da_dq[:,c,ind] += self.mxS(S,dv_dq[:,c,ind],qd[ind])
                if parent_ind != -1: # note that a_base is just gravity
                    da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,a[:,parent_ind]))
                else:
                    da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,gravity_vec))
                # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
                Imat = self.robot.get_Imat_by_id(ind)
                df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])
                Iv = np.matmul(Imat,v[:,ind])
                for c in range(n):
                    df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                    df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))

        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):

        if(self.overloading):
            # allocate memory
            n = len(qd)
            dv_dqd = matrix_(np.zeros((6,n,n)))
            da_dqd = matrix_(np.zeros((6,n,n)))
            df_dqd = matrix_(np.zeros((6,n,n)))

            # forward pass
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                S = matrix_(self.robot.get_S_by_id(ind))
                # dv_du = X * dv_du_parent + (if c == ind){S}
                if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                    dv_dqd[:,:,ind] = Xmat@dv_dqd[:,:,parent_ind]
                dv_dqd[:,ind,ind] = dv_dqd[:,ind,ind]+S
                # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
                if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                    da_dqd[:,:,ind] = Xmat@da_dqd[:,:,parent_ind]
                for c in range(n):
                    da_dqd[:,c,ind] = da_dqd[:,c,ind]+self.mxS(S,dv_dqd[:,c,ind],qd[ind])
                da_dqd[:,ind,ind] = da_dqd[:,ind,ind]+self.mxS(S,v[:,ind])
                # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
                Imat = matrix_(self.robot.get_Imat_by_id(ind))
                df_dqd[:,:,ind] = Imat@da_dqd[:,:,ind]
                Iv = Imat@v[:,ind]
                for c in range(n):
                    df_dqd[:,c,ind] = df_dqd[:,c,ind]+self.fxv(dv_dqd[:,c,ind],Iv)
                    df_dqd[:,c,ind] = df_dqd[:,c,ind]+self.fxv(v[:,ind],Imat@dv_dqd[:,c,ind])
        else:
            # allocate memory
            n = len(qd)
            dv_dqd = np.zeros((6,n,n))
            da_dqd = np.zeros((6,n,n))
            df_dqd = np.zeros((6,n,n))

            # forward pass
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)
                # dv_du = X * dv_du_parent + (if c == ind){S}
                if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                    dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
                dv_dqd[:,ind,ind] += S
                # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
                if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                    da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
                for c in range(n):
                    da_dqd[:,c,ind] += self.mxS(S,dv_dqd[:,c,ind],qd[ind])
                da_dqd[:,ind,ind] += self.mxS(S,v[:,ind])
                # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
                Imat = self.robot.get_Imat_by_id(ind)
                df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
                Iv = np.matmul(Imat,v[:,ind])
                for c in range(n):
                    df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                    df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))

        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):

        if(self.overloading):
            # allocate memory
            n = len(q) # assuming len(q) = len(qd)
            dc_dq = matrix_(np.zeros((n,n)))
            
            for ind in range(n-1,-1,-1):
                # dc_du is S^T*df_du
                S = matrix_(self.robot.get_S_by_id(ind))
                dc_dq[ind,:] = S.T@df_dq[:,:,ind]
                # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat =matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                    df_dq[:,:,parent_ind] = df_dq[:,:,parent_ind]+(Xmat.T@df_dq[:,:,ind])
                    delta_dq = Xmat.T@self.fxS(S,f[:,ind])
                    for entry in range(6):
                        df_dq[entry,ind,parent_ind] = df_dq[entry,ind,parent_ind]+delta_dq[entry]
        else:
            # allocate memory
            n = len(q) # assuming len(q) = len(qd)
            dc_dq = np.zeros((n,n))
            
            for ind in range(n-1,-1,-1):
                # dc_du is S^T*df_du
                S = self.robot.get_S_by_id(ind)
                dc_dq[ind,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
                # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                    df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                    delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                    for entry in range(6):
                        df_dq[entry,ind,parent_ind] += delta_dq[entry]

        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING = False):
        
        if(self.overloading):
            # allocate memory
            n = len(q) # assuming len(q) = len(qd)
            dc_dqd = matrix_(np.zeros((n,n)))
            
            for ind in range(n-1,-1,-1):
                # dc_du is S^T*df_du
                S = matrix_(self.robot.get_S_by_id(ind))
                dc_dqd[ind,:] = S.T@df_dqd[:,:,ind]
                # df_du_parent += X^T*df_du
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                    df_dqd[:,:,parent_ind] = df_dqd[:,:,parent_ind]+(Xmat.T@df_dqd[:,:,ind])
        else:
            # allocate memory
            n = len(q) # assuming len(q) = len(qd)
            dc_dqd = np.zeros((n,n))
            
            for ind in range(n-1,-1,-1):
                # dc_du is S^T*df_du
                S = self.robot.get_S_by_id(ind)
                dc_dqd[ind,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
                # df_du_parent += X^T*df_du
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                    df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind]) 

        # add in the damping
        if USE_VELOCITY_DAMPING:
            for ind in range(n):
                dc_dqd[ind,ind] += self.robot.get_damping_by_id(ind)

        return dc_dqd

    def rnea_grad(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):

        # instead of passing in trajectory, what if we want our planning algorithm to solve for the optimal trajectory?
        """
        The gradients of inverse dynamics can be very extremely useful inputs into trajectory optimization algorithmss.
        Input: trajectory, including position, velocity, and acceleration
        Output: Computes the gradient of joint forces with respect to the positions and velocities. 
        """ 
        
        (c, v, a, f) = self.rnea(q, qd, qdd, GRAVITY)

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)

        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

        if(self.overloading):
            dc_du = matrix_.hstack(dc_dq,dc_dqd)
        else:
            dc_du = np.hstack((dc_dq,dc_dqd))

        return dc_du

    
    def minv_bpass(self, q):

        if(self.overloading):
            # allocate memory
            n = len(q)
            Minv = matrix_(np.zeros((n,n)))
            F = matrix_(np.zeros((n,6,n)))
            U = matrix_(np.zeros((n,6)))
            Dinv = matrix_(np.zeros(n))

            # set initial IA to I
            IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())
            
            # backward pass
            for ind in range(n-1,-1,-1):
                # Compute U, D
                S = matrix_(self.robot.get_S_by_id(ind))
                subtreeInds = self.robot.get_subtree_by_id(ind)
                U[ind,:] = matrix_(IA[ind])@S
                Dinv[ind] = 1/(S.T@U[ind,:])
                # Update Minv
                Minv[ind,ind] = Dinv[ind]
                for subInd in subtreeInds:
                    Minv[ind,subInd] = Minv[ind,subInd]-(Dinv[ind]*(S.T@F[ind,:,subInd]))
                # update parent if applicable
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                    # update F
                    for subInd in subtreeInds:
                        F[ind,:,subInd] = F[ind,:,subInd]+(U[ind,:]*Minv[ind,subInd])
                        F[parent_ind,:,subInd] = F[parent_ind,:,subInd]+(Xmat.T@F[ind,:,subInd])
                    # update IA
                    Ia = matrix_(IA[ind])-matrix_(np.outer(U[ind,:],Dinv[ind]*U[ind,:]))
                    IaParent = Xmat.T@(Ia@Xmat)
                    IA[parent_ind] = IA[parent_ind]+IaParent
        else:
            # allocate memory
            n = len(q)
            Minv = np.zeros((n,n))
            F = np.zeros((n,6,n))
            U = np.zeros((n,6))
            Dinv = np.zeros(n)

            # set initial IA to I
            IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())
            
            # backward pass
            for ind in range(n-1,-1,-1):
                # Compute U, D
                S = self.robot.get_S_by_id(ind)
                subtreeInds = self.robot.get_subtree_by_id(ind)
                U[ind,:] = np.matmul(IA[ind],S)
                Dinv[ind] = 1/np.matmul(S.transpose(),U[ind,:])
                # Update Minv
                Minv[ind,ind] = Dinv[ind]
                for subInd in subtreeInds:
                    Minv[ind,subInd] -= Dinv[ind] * np.matmul(S.transpose(),F[ind,:,subInd])
                # update parent if applicable
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                    # update F
                    for subInd in subtreeInds:
                        F[ind,:,subInd] += U[ind,:]*Minv[ind,subInd]
                        F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[ind,:,subInd]) 
                    # update IA
                    Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
                    IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
                    IA[parent_ind] += IaParent

        return (Minv, F, U, Dinv)

    def minv_fpass(self, q, Minv, F, U, Dinv):

        n = len(q)
        if(self.overloading):
            # forward pass
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                S = matrix_(self.robot.get_S_by_id(ind))
                Xmat = matrix_(self.robot.get_Xmat_Func_by_id(ind)(q[ind]))
                if parent_ind != -1:
                    Minv[ind,ind:] = Minv[ind,ind:]-(Dinv[ind]*((U[ind,:].T@Xmat)@F[parent_ind,:,ind:]))
                
                F[ind,:,ind:] = matrix_(np.outer(S,Minv[ind,ind:]))
                if parent_ind != -1:
                    F[ind,:,ind:] = F[ind,:,ind:]+(Xmat@F[parent_ind,:,ind:])
        else:
            # forward pass
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                S = self.robot.get_S_by_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                if parent_ind != -1:
                    Minv[ind,ind:] -= Dinv[ind]*np.matmul(np.matmul(U[ind,:].transpose(),Xmat),F[parent_ind,:,ind:])
                
                F[ind,:,ind:] = np.outer(S,Minv[ind,ind:])
                if parent_ind != -1:
                    F[ind,:,ind:] += np.matmul(Xmat,F[parent_ind,:,ind:])

        return Minv

    def minv(self, q, output_dense = True):
        # based on https://www.researchgate.net/publication/rnea343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
        """ Computes the analytical inverse of the joint space inertia matrix
        CRBA calculates the joint space inertia matrix H to represent the composite inertia
        This is used in the fundamental motion equation H qdd + C = Tau
        Forward dynamics roughly calculates acceleration as H_inv ( Tau - C); analytic inverse - benchmark against Pinocchio
        """
        # backward pass
        (Minv, F, U, Dinv) = self.minv_bpass(q)

        # forward pass
        Minv = self.minv_fpass(q, Minv, F, U, Dinv)

        # fill in full matrix (currently only upper triangular)
        if output_dense:
            n = len(q)
            for col in range(n):
                for row in range(n):
                    if col < row:
                        Minv[row,col] = Minv[col,row]

        self.saved_minv.append(Minv)
        return Minv
   
   