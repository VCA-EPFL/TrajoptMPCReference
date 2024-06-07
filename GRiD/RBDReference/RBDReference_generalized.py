import numpy as np
import copy
import sympy as sp
np.set_printoptions(precision=4, suppress=True, linewidth = 100)
import time
class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj # instance of Robot Object class created by URDFparser``

    def cross_operator(self, v):
        # for any vector v, computes the operator v x 
        # vec x = [wx   0]
        #         [vox wx]
        #(crm in spatial_v2_extended)
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
        return -np.asmatrix(res)


    def mxS(self, S, vec, alpha=1.0):
        # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
        # derivative of spatial motion vector = v x m
        # return(alpha * np.dot(self.cross_operator(vec), S))     
        return(alpha * self.cross_operator(vec).dot(S))    
    
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

        eePos_arr = []
        for jid in self.robot.get_leaf_nodes():
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            Xmat_hom = np.eye(4)
            for ind in jidChain:
                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                Xmat_hom = np.matmul(Xmat_hom,currX)

            currId = jid
            Xmat_hom = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
            currId = self.robot.get_parent_id(currId)
            while(currId != -1):
                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
                Xmat_hom = np.matmul(currX,Xmat_hom)
                currId = self.robot.get_parent_id(currId)

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

        print("Sym Jacobian\n", J)
        return J
        
    #NOT USED
    def jacobian_grad_func(self,offsets = [np.matrix([[0,1,0,1]])]):
        n = self.robot.get_num_joints()
        q_symbols = [sp.Symbol('q{}'.format(i)) for i in range(1, n+1)]
        J=self.symbolic_jacobian(offsets)
        #need symbolic expression to diff
        dJdq = [[sp.diff(J[i, j], q) for q in q_symbols] for i in range(J.shape[0]) for j in range(J.shape[1])]
        print("dJdq sym;\n",dJdq)
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
        # print("dJdq 1\n", dJdq)

        n = self.robot.get_num_joints()
        dJdq=np.zeros((2*n,n))
        J=self.Jacobian(q,offsets)

        if n==2:
            dJdq[0,:] = -J[1,:]
            dJdq[1,:] = [-J[1,1],-J[1,1]]
            dJdq[2,:] = J[0,:]
            dJdq[3,:] = [J[0,1],J[0,1]]

        if n==3:
            dJdq[0,:] = -J[1,:]
            dJdq[1,:2]= [-J[1,1],-J[1,1]]
            dJdq[1,2] = -J[1,2]
            dJdq[2,:] = [-J[1,2]]*n
            dJdq[3,:] = J[0,:]
            dJdq[4,:2]= [J[0,1],J[0,1]]
            dJdq[4,2] = J[0,2]
            dJdq[5,:] = [J[0,2]]*n
        # Need to do generalize for every n
        return dJdq


    # Commented part => product of Xmat and ddXmat
    # dJdq has same elements as J => less compute
    def d2Jdq2(self,q ,offsets = [np.matrix([[0,1,0,1]])]):
        n = self.robot.get_num_joints()
        ddJdq=np.zeros((2*n,n))
        J_list = []
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
                    
                        if ind == dind: # use second derivative
                            currX = self.robot.get_dddXmat_hom_Func_by_id(ind)(q[ind])
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                        Xmat_hom = np.matmul(Xmat_hom,currX)
                    deePos_xyz1 = Xmat_hom * offsets[0].transpose()
                    deePos_col = deePos_xyz1[:3,:]
                    deePos = self.equals_or_hstack(deePos,deePos_col)

            J_list.append(deePos)
        ddJdq[0,:]= J_list[0][0,:n]
        ddJdq[1,:]= [ddJdq[0,1]]*n
        ddJdq[2,:]= J_list[0][1,:n]
        ddJdq[3,:]= [ddJdq[2,1]]*n


        n = self.robot.get_num_joints()
        ddJdq=np.zeros((2*n,n))
        J=self.Jacobian(q,offsets) # compute J twice (in dJdq and ddJdq)
        if n==2:
            ddJdq[0,:] = -J[0,:]
            ddJdq[1,:] = [-J[0,1],-J[0,1]]
            ddJdq[2,:] = -J[1,:]
            ddJdq[3,:] = [-J[1,1],-J[1,1]]

        if n==3:
            ddJdq[0,:] = -J[0,:]
            ddJdq[1,:2]= [-J[0,1],-J[0,1]]
            ddJdq[1,2] = -J[0,2]
            ddJdq[2,:] = [-J[0,2]]*n
            ddJdq[3,:] = J[1,:]
            ddJdq[4,:2]= [J[1,1],J[1,1]]
            ddJdq[4,2] = J[1,2]
            ddJdq[5,:] = [J[1,2]]*n

        return ddJdq

    #NOT USED
    # def jacobian_grad(self,q):
    #  [q1,q2]=q
    #  s1=np.sin(q1)
    #  s2=np.sin(q2)
    #  c1=np.cos(q1)
    #  c2=np.cos(q2)
    #  return np.array([[s1*c2 + s1 + s2*c1, s1*c2 + s2*c1],\
    #                    [s1*c2 + s2*c1, s1*c2 + s2*c1], \
    #                    [s1*s2 - c1*c2 - c1, s1*s2 - c1*c2], \
    #                    [s1*s2 - c1*c2, s1*s2 - c1*c2]])
    
    #d(x,y,vx,vy)/d(q,q_d)
    def jacobian_tot_state(self,q,qd,offsets = [np.matrix([[0,1,0,1]])]):
        n = self.robot.get_num_joints()
        J1=self.Jacobian(q,offsets)
        dJdq=self.dJdq(q,offsets)
        J2=(dJdq@qd).reshape(2,n)
        J_top = np.hstack((J1, np.zeros_like(J1)))
        J_bottom = np.hstack((J2, J1))
        J = np.vstack((J_top, J_bottom))
        return J
    
    #d(x,y)/dq
    def Jacobian(self, q, offsets = [np.matrix([[0,1,0,1]])]):
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
        return jacobian[0][:2,:n]

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
               # print("RNEA v:", v[:,parent_ind])
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

        dc_du = np.hstack((dc_dq,dc_dqd))
        return dc_du

    

    def idsva(self, q, qd, qdd, GRAVITY = -9.81):
        """alternative to rnea_grad(), described in "Efficient Analytical Derivatives of Rigid-Body Dynamics using
Spatial Vector Algebra" (Singh, Russel, and Wensing)

        :param q: initial joint positions
        :type q: array representing a 1 by n vector, where n is the number of joints
        :param qd: initial joint velocities
        :type qd: array representing a 1 by n vector
        :param qdd: desired joint accelerations 
        :type qdd: array representing a 1 by n vector
        :param GRAVITY: defaults to -9.81
        :type GRAVITY: float, optional
        :return: dtau_dq, dtau_dqd-- the gradient of the resulting torque with respect to initial joint position and velocity
        :rtype: list
        """
        # allocate memory
        n = len(qd)
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        Xup0 =  [None] * n #list of transformation matrices in the world frame 
        Xdown0 = [None] * n
        S = np.zeros((6,n))
        Sd = np.zeros((6,n))
        Sdd = np.zeros((6,n))
        Sj = np.zeros((6,n)) 
        IC = [None] * n 
        BC  = [None] * n 
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        
        # forward pass
        for i in range(n):
            parent_i = self.robot.get_parent_id(i)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])

            # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                a[:,i] = Xmat @ gravity_vec
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i])

            S[:,i] = self.robot.get_S_by_id(i)

            S[:,i] = Xdown0[i] @ S[:,i]
            Sd[:,i] = self.cross_operator(v[:,i]) @ S[:,i]
            Sdd[:,i]= self.cross_operator(a[:,i])@ S[:,i]
            Sdd[:,i] = Sdd[:,i] + self.cross_operator(v[:,i])@ Sd[:,i]
            Sj[:,i] = 2*Sd[:,i] + self.cross_operator(S[:,i]*qd[i])@ S[:,i]

            v6x6 = self.cross_operator(v[:,i])
            v[:,i] = v[:,i] + S[:,i]*qd[i]
            a[:,i] = a[:,i] + np.array(v6x6 @ S[:,i]*qd[i])

            if qdd is not None:
                a[:,i] += S[:,i]*qdd[i]
            
            # compute f, IC, BC
            Imat = self.robot.get_Imat_by_id(i)

            IC[i] = np.array(Xup0[i]).T  @ (Imat @ Xup0[i])
            f[:,i] = IC[i] @ a[:,i] + self.dual_cross_operator(v[:,i]) @ IC[i] @ v[:,i]
            f[:,i] = np.asarray(f[:,i]).flatten()
            BC[i] = (self.dual_cross_operator(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.cross_operator(v[:,i]))
        

        t1 = np.zeros((6,n))
        t2 = np.zeros((6,n))
        t3 = np.zeros((6,n))
        t4 = np.zeros((6,n))
        dtau_dq = np.zeros((n,n))
        dtau_dqd = np.zeros((n,n))


        #backward pass
        for i in range(n-1,-1,-1):

            t1[:,i] = IC[i] @ S[:,i]
            t2[:,i] = BC[i] @ S[:,i].T + IC[i] @ Sj[:,i]
            t3[:,i] = BC[i] @ Sd[:,i] + IC[i] @ Sdd[:,i] + self.icrf(f[:,i]) @ S[:,i]
            t4[:,i] = BC[i].T @ S[:,i]

            subtree_ids = self.robot.get_subtree_by_id(i) #list of all subtree ids (inclusive)



            dtau_dq[i, subtree_ids[1:]] = S[:,i] @ t3[:,subtree_ids[1:]]
            
            dtau_dq[subtree_ids[0:], i] = Sdd[:,i] @ t1[:,subtree_ids[0:]] + \
                                        Sd[:,i] @ t4[:,subtree_ids[0:]] 
            
            dtau_dqd[i, subtree_ids[1:]] = S[:,i] @ t2[:,subtree_ids[1:]]

            dtau_dqd[subtree_ids[0:], i] = Sj[:,i] @ t1[:,subtree_ids[0:]] + \
                                        S[:,i] @ t4[:,subtree_ids[0:]] 

            p = self.robot.get_parent_id(i)
            if p >= 0:
                IC[p] = IC[p] + IC[i]
                BC[p] = BC[p] + BC[i]
                f[:,p] = f[:,p] + f[:,i]


        return dtau_dq, dtau_dqd


    def minv_bpass(self, q):
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

        return Minv

    def crm(v):
        if len(v) == 6:
            vcross = np.array([0, -v[3], v[2], 0,0,0], [v[3], 0, -v[1], 0,0,0], [-v[2], v[1], 0, 0,0,0], [0, -v[6], v[5], 0,-v[3],v[2]], [v[6], 0, -v[4], v[3],0,-v[1]], [-v[5], v[4], 0, -v[2],v[1],0])
        else:
            vcross = np.array([0, 0, 0], [v[3], 0, -v[1]], [-v[2], v[1], 0])
        return vcross

    def aba(self, q, qd, tau, GRAVITY = -9.81):
       # allocate memory
        n = len(qd)
        v = np.zeros((6,n))
        c = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        d = np.zeros(n)
        U = np.zeros((6,n))
        u = np.zeros(n)
        IA = np.zeros((6,6,n))
        pA = np.zeros((6,n))
        qdd = np.zeros(n)
        
        
        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec
                 
        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)

            if parent_ind == -1: # parent is base
                v[:,ind] = S*qd[ind]
                
            else:
                v[:,ind] = np.matmul(Xmat,v[:,parent_ind])
                v[:,ind] += S*qd[ind]
                c[:,ind] = self.mxS(S,v[:,ind],qd[ind])

            Imat = self.robot.get_Imat_by_id(ind)

            IA[:,:,ind] = Imat

            vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
            [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
            [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
            [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
            [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
            [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

            crf=-np.transpose(vcross)
            temp=np.matmul(crf,Imat)

            pA[:,ind]=np.matmul(temp,v[:,ind])[0]
        
        for ind in range(n-1,-1,-1):
            S = self.robot.get_S_by_id(ind)
            parent_ind = self.robot.get_parent_id(ind)

            U[:,ind] = np.matmul(IA[:,:,ind],S)
            d[ind] = np.matmul(np.transpose(S),U[:,ind])
            u[ind] = tau[ind] - np.matmul(np.transpose(S),pA[:,ind])

            if parent_ind != -1:

                rightSide=np.reshape(U[:,ind],(6,1))@np.reshape(U[:,ind],(6,1)).T/d[ind]
                Ia = IA[:,:,ind] - rightSide

                pa = pA[:,ind] + np.matmul(Ia, c[:,ind]) + U[:,ind]*u[ind]/d[ind]

                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                temp = np.matmul(np.transpose(Xmat), Ia)

                IA[:,:,parent_ind] = IA[:,:,parent_ind] + np.matmul(temp,Xmat)

                temp = np.matmul(np.transpose(Xmat), pa)
                pA[:,parent_ind]=pA[:,parent_ind] + temp.flatten()
                                             
        for ind in range(n):

            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

            if parent_ind == -1: # parent is base
                a[:,ind] = np.matmul(Xmat,gravity_vec) + c[:,ind]
            else:
                a[:,ind] = np.matmul(Xmat, a[:,parent_ind]) + c[:,ind]

            S = self.robot.get_S_by_id(ind)
            temp = u[ind] - np.matmul(np.transpose(U[:,ind]),a[:,ind])
            qdd[ind] = temp / d[ind]
            a[:,ind] = a[:,ind] + qdd[ind]*S
        
        return qdd
    
    def crba( self, q, qd, tau):
        n = len(qd)
        
        C = self.rnea(q, qd, qdd = None, GRAVITY = -9.81)[0]

        IC = copy.deepcopy(self.robot.get_Imats_dict_by_id())# composite inertia calculation

        for ind in range(n-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

            if parent_ind != -1:
                IC[parent_ind] = IC[parent_ind] + np.matmul(np.matmul(Xmat.T, IC[ind]),Xmat);

        H = np.zeros((n,n))

        for ind in range(n):

            S = self.robot.get_S_by_id(ind)
            fh = np.matmul(IC[ind],S)
            H[ind,ind] = np.matmul(S,fh)
            j = ind;

            while self.robot.get_parent_id(j) > -1:
                Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
                fh = np.matmul(Xmat.T,fh);
                j = self.robot.get_parent_id(j)
                S = self.robot.get_S_by_id(j)
                H[ind,j] = np.matmul(S.T, fh);
                H[j,ind] = H[ind,j];

        sub=np.subtract(tau,C)
        return H

    

# import numpy as np
# import copy
# np.set_printoptions(precision=4, suppress=True, linewidth = 100)

# class RBDReference:
#     def __init__(self, robotObj):
#         self.robot = robotObj # instance of Robot Object class created by URDFparser``

#     def cross_operator(self, v):
#         # for any vector v, computes the operator v x 
#         # vec x = [wx   0]
#         #         [vox wx]
#         #(crm in spatial_v2_extended)
#         v_cross = np.array([0, -v[2], v[1], 0, 0, 0,
#                             v[2], 0, -v[0], 0, 0, 0,
#                             -v[1], v[0], 0, 0, 0, 0,
#                             0, -v[5], v[4], 0, -v[2], v[1], 
#                             v[5], 0, -v[3], v[2], 0, -v[0],
#                             -v[4], v[3], 0, -v[1], v[0], 0]
#                           ).reshape(6,6)
#         return(v_cross)
    
#     def dual_cross_operator(self, v):
#         #(crf in in spatial_v2_extended)
#         return(-1 * self.cross_operator(v).T)
    

#     def icrf(self, v):
#         #helper function defined in spatial_v2_extended library, called by idsva()
#         res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
#             [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
#             [-v[1],  v[0],    0,  -v[4],  v[3],    0],
#             [    0,  -v[5],  v[4],    0,    0,    0],
#             [ v[5],    0,  -v[3],    0,    0,    0],
#             [-v[4],  v[3],    0,    0,    0,    0]]
#         return -np.asmatrix(res)


#     def mxS(self, S, vec, alpha=1.0):
#         # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
#         # derivative of spatial motion vector = v x m
#         return(alpha * np.dot(self.cross_operator(vec), S))        
    
#     def fxv_simple(self, m, f):
#         # force spatial vector cross product. 
#         return(np.dot(self.dual_cross_operator(m), f))

#     def fxv(self, fxVec, timesVec):
#         # Fx(fxVec)*timesVec
#         #   0  -v(2)  v(1)    0  -v(5)  v(4)
#         # v(2)    0  -v(0)  v(5)    0  -v(3)
#         #-v(1)  v(0)    0  -v(4)  v(3)    0
#         #   0     0     0     0  -v(2)  v(1)
#         #   0     0     0   v(2)    0  -v(0)
#         #   0     0     0  -v(1)  v(0)    0
#         result = np.zeros((6))
#         result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2] - fxVec[5] * timesVec[4] + fxVec[4] * timesVec[5]
#         result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2] + fxVec[5] * timesVec[3] - fxVec[3] * timesVec[5]
#         result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1] - fxVec[4] * timesVec[3] + fxVec[3] * timesVec[4]
#         result[3] =                                                     -fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
#         result[4] =                                                      fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
#         result[5] =                                                     -fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
#         return result

#     def fxS(self, S, vec, alpha = 1.0):
#         # force spatial cross product with motion subspace 
#         return -self.mxS(S, vec, alpha)

#     def vxIv(self, vec, Imat):
#         # necessary component in differentiating Iv (product rule).
#         # We express I_dot x v as v x (Iv) (see Featherstone 2.14)
#         # our core equation of motion is f = d/dt (Iv) = Ia + vx* Iv
#         temp = np.matmul(Imat,vec)
#         vecXIvec = np.zeros((6))
#         vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
#         vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
#         vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
#         vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
#         vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
#         vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
#         return vecXIvec

#     """
#     End Effector Posiitons

#     offests is an array of np matricies of the form (offset_x, offset_y, offset_z, 1)
#     """
#     def end_effector_positions(self, q, offsets = [np.matrix([[0,0,0,1]])]):

#         # do for each branch in the chain
#         eePos_arr = []
#         for jid in self.robot.get_leaf_nodes():
            
#             # chain up the transforms (version 1 for starting from the root)
#             # first get the joints in the chain
#             jidChain = sorted(self.robot.get_ancestors_by_id(jid))
#             jidChain.append(jid)
#             # then chain them up
#             Xmat_hom = np.eye(4)
#             for ind in jidChain:
#                 currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
#                 Xmat_hom = np.matmul(Xmat_hom,currX)

#             # chain up the transforms (version 2 for starting from the leaf)
#             currId = jid
#             Xmat_hom = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
#             currId = self.robot.get_parent_id(currId)
#             while(currId != -1):
#                 currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
#                 Xmat_hom = np.matmul(currX,Xmat_hom)
#                 currId = self.robot.get_parent_id(currId)

#             # Then extract the end-effector position with the given offset(s)
#             # TODO update for multiple offsets

#             # xyz position is easy
#             eePos_xyz1 = Xmat_hom * offsets[0].transpose()

#             # roll pitch yaw is a bit more difficult
#             eePos_roll = np.arctan2(Xmat_hom[2,1],Xmat_hom[2,2])
#             pitch_temp = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
#             eePos_pitch = np.arctan2(-Xmat_hom[2,0],pitch_temp)
#             eePos_yaw = np.arctan2(Xmat_hom[1,0],Xmat_hom[0,0])
#             eePos_rpy = np.matrix([[eePos_roll,eePos_pitch,eePos_yaw]])

#             # then stack it up!
#             eePos = np.vstack((eePos_xyz1[:3,:],eePos_rpy.transpose()))
#             eePos_arr.append(eePos)
#         return eePos_arr

#     """
#     End Effectors Position Gradients
#     """
#     def equals_or_hstack(self, obj, col):
#         if obj is None:
#             obj = col
#         else:
#             obj = np.hstack((obj,col))
#         return obj
#     def end_effector_position_gradients(self, q, offsets = [np.matrix([[0,0,0,1]])]):
#         n = self.robot.get_num_joints()
        
#         # For each branch chain up the transformations across all possible derivatives
#         # Note: if not in branch then 0
#         deePos_arr = []
#         for jid in self.robot.get_leaf_nodes():
            
#             # first get the joints in the chain
#             jidChain = sorted(self.robot.get_ancestors_by_id(jid))
#             jidChain.append(jid)

#             # then compute the gradients
#             deePos = None
#             for dind in range(n):

#                 # Note: if not in branch then 0
#                 if dind not in jidChain:
#                     deePos_col = np.zeros((6,1))
#                     deePos = self.equals_or_hstack(deePos,deePos_col)
                
#                 else:
#                     # first chain up the transforms
#                     Xmat_hom = np.eye(4)
#                     for ind in jidChain:
#                         if ind == dind: # use derivative
#                             currX = self.robot.get_dXmat_hom_Func_by_id(ind)(q[ind])
#                         else: # use normal transform
#                             currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
#                         Xmat_hom = np.matmul(Xmat_hom,currX)

#                     # chain up the transforms (version 2 for starting from the leaf)
#                     currId = jid
#                     if currId == dind:
#                         Xmat_hom = self.robot.get_dXmat_hom_Func_by_id(currId)(q[currId])
#                     else:
#                         Xmat_hom = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
#                     currId = self.robot.get_parent_id(currId)
#                     while(currId != -1):
#                         if currId == dind:
#                             currX = self.robot.get_dXmat_hom_Func_by_id(currId)(q[currId])
#                         else:
#                             currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
#                         Xmat_hom = np.matmul(currX,Xmat_hom)
#                         currId = self.robot.get_parent_id(currId)
                    
#                     # Then extract the end-effector position with the given offset(s)
#                     # TODO handle different offsets for different branches

#                     # xyz position is easy
#                     deePos_xyz1 = Xmat_hom * offsets[0].transpose()

#                     # roll pitch yaw is a bit more difficult
#                     # TODO THESE ARE WRONG BECUASE THERE IS CHAIN RULE HERE
#                     # see https://github.com/plancherb1/parallel-DDP/blob/master/plants/dynamics_arm.cuh
#                     # but note the mistake on indexing -- these are the corret indexes
#                     deePos_roll = np.arctan2(Xmat_hom[2,1],Xmat_hom[2,2])
#                     pitch_temp = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
#                     deePos_pitch = np.arctan2(-Xmat_hom[2,0],pitch_temp)
#                     deePos_yaw = np.arctan2(Xmat_hom[1,0],Xmat_hom[0,0])
#                     deePos_rpy = np.matrix([[deePos_roll,deePos_pitch,deePos_yaw]])

#                     # then stack it up!
#                     deePos_col = np.vstack((deePos_xyz1[:3,:],deePos_rpy.transpose()))
#                     deePos = self.equals_or_hstack(deePos,deePos_col)

#             deePos_arr.append(deePos)
#         return deePos_arr

#     """
#     Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

#     RNEA divided into 3 parts: 
#         1) calculate the velocity and acceleration of each body in the tree
#         2) Calculate the forces necessary to produce these accelertions
#         3) Calculate the forces transmitted across the joints from the forces acting on the bodies
#     """

    
#     def rnea_fpass(self, q, qd, qdd = None, GRAVITY = -9.81):
#         """
#         Forward Pass for RNEA algorithm. Computes the velocity and acceleration of each body in the tree necessary to produce a certain trajectory
        
#         OUTPUT: 
#         v : input qd is specifying value within configuration space with assumption of one degree of freedom. 
#         Output velocity is in general body coordinates and specifies motion in full 6 degrees of freedom
#         """
#         assert len(q) == len(qd), "Invalid Trajectories"
#         # not sure should equal num links or num joints. 
#         assert len(q) == self.robot.get_num_joints(), "Invalid Trajectory, must specify coordinate for every body" 
#         # allocate memory
#         n = len(q)
#         v = np.zeros((6,n))
#         a = np.zeros((6,n))
#         f = np.zeros((6,n))

#         gravity_vec = np.zeros((6)) # model gravity as a fictitious base acceleration. 
#         # all forces subsequently offset by gravity. 
#         gravity_vec[5] = -GRAVITY # a_base is gravity vec, linear in z direction

#         # forward pass
#         # vi = vparent + Si * qd_i
#         # differentiate for ai = aparent + Si * qddi + Sdi * qdi
        
#         for ind in range(n):
#             parent_ind = self.robot.get_parent_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind]) # the coordinate transform that brings into base reference frame
#             S = self.robot.get_S_by_id(ind) # Subspace matrix
#             # compute v and a
#             if parent_ind == -1: # parent is base
#                 # v_base is zero so v[:,ind] remains 0
#                 a[:,ind] = np.matmul(Xmat,gravity_vec)
#             else:
#                 v[:,ind] = np.matmul(Xmat,v[:,parent_ind]) # velocity of parent in base coordinates. 
#                 a[:,ind] = np.matmul(Xmat,a[:,parent_ind])
#             v[:,ind] += S*qd[ind] # S turns config space into actual velocity
#             a[:,ind] += self.mxS(S,v[:,ind],qd[ind])
#             if qdd is not None:
#                 a[:,ind] += S*qdd[ind]

#             # compute f
#             Imat = self.robot.get_Imat_by_id(ind)
#             f[:,ind] = np.matmul(Imat,a[:,ind]) + self.vxIv(v[:,ind],Imat)

#         return (v,a,f)

#     def rnea_bpass(self, q, qd, f, USE_VELOCITY_DAMPING = False):
#         # allocate memory
#         n = len(q) # assuming len(q) = len(qd)
#         c = np.zeros(n)
        
#         # backward pass
#         # seek to calculate force transmitted from body i across joint i (fi) from the outside in.
#         # fi = fi^B (net force) - fi^x (external forces, assumed to be known) - sum{f^j (all forces from children)}. 
#         # Start with outermost as set of children is empty and go backwards to base.
#         for ind in range(n-1,-1,-1):
#             S = self.robot.get_S_by_id(ind)
#             # compute c
#             c[ind] = np.matmul(np.transpose(S),f[:,ind])
#             # update f if applicable
#             parent_ind = self.robot.get_parent_id(ind)
#             if parent_ind != -1:
#                 Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#                 temp = np.matmul(np.transpose(Xmat),f[:,ind])
#                 f[:,parent_ind] = f[:,parent_ind] + temp.flatten()

#         # add velocity damping (defaults to 0)
#         if USE_VELOCITY_DAMPING:
#             for k in range(n):
#                 c[k] += self.robot.get_damping_by_id(k) * qd[k]

#         return (c,f)

#     def rnea(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
#         """
#         Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

#         RNEA divided into 3 parts: 
#             1) calculate the velocity and acceleration of each body in the tree
#             2) Calculate the forces necessary to produce these accelertions
#             3) Calculate the forces transmitted across the joints from the forces acting on the bodies
            
#         INPUT:
#         q, qd, qdd: position, velocity, acceleration. Nx1 arrays where N is the number of bodies
#         GRAVITY - gravitational field of the body; default is earth surface gravity, 9.81
#         USE_VELOCITY_DAMPING: flag for whether velocity is damped, representing ___
        
#         OUTPUTS: 
#         c: Coriolis terms and other forces potentially be applied to the system. 
#         v: velocity of each joint in world base coordinates rather than motion subspace
#         a: acceleration of each joint in world base coordinates rather than motion subspace
#         f: forces that joints must apply to produce trajectory
#         """
#         # forward pass
#         (v,a,f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
#         # backward pass
#         (c,f) = self.rnea_bpass(q, qd, f, USE_VELOCITY_DAMPING)

#         return (c,v,a,f)

#     def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        
#         # allocate memory
#         n = len(qd)
#         dv_dq = np.zeros((6,n,n))
#         da_dq = np.zeros((6,n,n))
#         df_dq = np.zeros((6,n,n))

#         gravity_vec = np.zeros((6))
#         gravity_vec[5] = -GRAVITY # a_base is gravity vec

#         for ind in range(n):
#             parent_ind = self.robot.get_parent_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#             S = self.robot.get_S_by_id(ind)
#             # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
#             if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
#                 dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
#                 dv_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,v[:,parent_ind]))
#             # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
#             if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
#                 da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
#             for c in range(n):
#                 da_dq[:,c,ind] += self.mxS(S,dv_dq[:,c,ind],qd[ind])
#             if parent_ind != -1: # note that a_base is just gravity
#                 da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,a[:,parent_ind]))
#             else:
#                 da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,gravity_vec))
#             # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
#             Imat = self.robot.get_Imat_by_id(ind)
#             df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])
#             Iv = np.matmul(Imat,v[:,ind])
#             for c in range(n):
#                 df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
#                 df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))

#         return (dv_dq, da_dq, df_dq)

#     def rnea_grad_fpass_dqd(self, q, qd, v):
        
#         # allocate memory
#         n = len(qd)
#         dv_dqd = np.zeros((6,n,n))
#         da_dqd = np.zeros((6,n,n))
#         df_dqd = np.zeros((6,n,n))

#         # forward pass
#         for ind in range(n):
#             parent_ind = self.robot.get_parent_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#             S = self.robot.get_S_by_id(ind)
#             # dv_du = X * dv_du_parent + (if c == ind){S}
#             if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
#                 dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
#             dv_dqd[:,ind,ind] += S
#             # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
#             if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
#                 da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
#             for c in range(n):
#                 da_dqd[:,c,ind] += self.mxS(S,dv_dqd[:,c,ind],qd[ind])
#             da_dqd[:,ind,ind] += self.mxS(S,v[:,ind])
#             # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
#             Imat = self.robot.get_Imat_by_id(ind)
#             df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
#             Iv = np.matmul(Imat,v[:,ind])
#             for c in range(n):
#                 df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
#                 df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))

#         return (dv_dqd, da_dqd, df_dqd)

#     def rnea_grad_bpass_dq(self, q, f, df_dq):
        
#         # allocate memory
#         n = len(q) # assuming len(q) = len(qd)
#         dc_dq = np.zeros((n,n))
        
#         for ind in range(n-1,-1,-1):
#             # dc_du is S^T*df_du
#             S = self.robot.get_S_by_id(ind)
#             dc_dq[ind,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
#             # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
#             parent_ind = self.robot.get_parent_id(ind)
#             if parent_ind != -1:
#                 Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#                 df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
#                 delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
#                 for entry in range(6):
#                     df_dq[entry,ind,parent_ind] += delta_dq[entry]

#         return dc_dq

#     def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING = False):
        
#         # allocate memory
#         n = len(q) # assuming len(q) = len(qd)
#         dc_dqd = np.zeros((n,n))
        
#         for ind in range(n-1,-1,-1):
#             # dc_du is S^T*df_du
#             S = self.robot.get_S_by_id(ind)
#             dc_dqd[ind,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
#             # df_du_parent += X^T*df_du
#             parent_ind = self.robot.get_parent_id(ind)
#             if parent_ind != -1:
#                 Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#                 df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind]) 

#         # add in the damping
#         if USE_VELOCITY_DAMPING:
#             for ind in range(n):
#                 dc_dqd[ind,ind] += self.robot.get_damping_by_id(ind)

#         return dc_dqd

#     def rnea_grad(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
#         # instead of passing in trajectory, what if we want our planning algorithm to solve for the optimal trajectory?
#         """
#         The gradients of inverse dynamics can be very extremely useful inputs into trajectory optimization algorithmss.
#         Input: trajectory, including position, velocity, and acceleration
#         Output: Computes the gradient of joint forces with respect to the positions and velocities. 
#         """ 
        
#         (c, v, a, f) = self.rnea(q, qd, qdd, GRAVITY)

#         # forward pass, dq
#         (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)

#         # forward pass, dqd
#         (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

#         # backward pass, dq
#         dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

#         # backward pass, dqd
#         dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

#         dc_du = np.hstack((dc_dq,dc_dqd))
#         return dc_du

    

#     def idsva(self, q, qd, qdd, GRAVITY = -9.81):
#         """alternative to rnea_grad(), described in "Efficient Analytical Derivatives of Rigid-Body Dynamics using
# Spatial Vector Algebra" (Singh, Russel, and Wensing)

#         :param q: initial joint positions
#         :type q: array representing a 1 by n vector, where n is the number of joints
#         :param qd: initial joint velocities
#         :type qd: array representing a 1 by n vector
#         :param qdd: desired joint accelerations 
#         :type qdd: array representing a 1 by n vector
#         :param GRAVITY: defaults to -9.81
#         :type GRAVITY: float, optional
#         :return: dtau_dq, dtau_dqd-- the gradient of the resulting torque with respect to initial joint position and velocity
#         :rtype: list
#         """
#         # allocate memory
#         n = len(qd)
#         v = np.zeros((6,n))
#         a = np.zeros((6,n))
#         f = np.zeros((6,n))
#         Xup0 =  [None] * n #list of transformation matrices in the world frame 
#         Xdown0 = [None] * n
#         S = np.zeros((6,n))
#         Sd = np.zeros((6,n))
#         Sdd = np.zeros((6,n))
#         Sj = np.zeros((6,n)) 
#         IC = [None] * n 
#         BC  = [None] * n 
#         gravity_vec = np.zeros(6)
#         gravity_vec[5] = -GRAVITY # a_base is gravity vec

        
#         # forward pass
#         for i in range(n):
#             parent_i = self.robot.get_parent_id(i)
#             Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])

#             # compute X, v and a
#             if parent_i == -1: # parent is base
#                 Xup0[i] = Xmat
#                 a[:,i] = Xmat @ gravity_vec
#             else:
#                 Xup0[i] = Xmat @ Xup0[parent_i]
#                 v[:,i] = v[:,parent_i]
#                 a[:,i] = a[:,parent_i]

#             Xdown0[i] = np.linalg.inv(Xup0[i])

#             S[:,i] = self.robot.get_S_by_id(i)

#             S[:,i] = Xdown0[i] @ S[:,i]
#             Sd[:,i] = self.cross_operator(v[:,i]) @ S[:,i]
#             Sdd[:,i]= self.cross_operator(a[:,i])@ S[:,i]
#             Sdd[:,i] = Sdd[:,i] + self.cross_operator(v[:,i])@ Sd[:,i]
#             Sj[:,i] = 2*Sd[:,i] + self.cross_operator(S[:,i]*qd[i])@ S[:,i]

#             v6x6 = self.cross_operator(v[:,i])
#             v[:,i] = v[:,i] + S[:,i]*qd[i]
#             a[:,i] = a[:,i] + np.array(v6x6 @ S[:,i]*qd[i])

#             if qdd is not None:
#                 a[:,i] += S[:,i]*qdd[i]
            
#             # compute f, IC, BC
#             Imat = self.robot.get_Imat_by_id(i)

#             IC[i] = np.array(Xup0[i]).T  @ (Imat @ Xup0[i])
#             f[:,i] = IC[i] @ a[:,i] + self.dual_cross_operator(v[:,i]) @ IC[i] @ v[:,i]
#             f[:,i] = np.asarray(f[:,i]).flatten()
#             BC[i] = (self.dual_cross_operator(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.cross_operator(v[:,i]))
        

#         t1 = np.zeros((6,n))
#         t2 = np.zeros((6,n))
#         t3 = np.zeros((6,n))
#         t4 = np.zeros((6,n))
#         dtau_dq = np.zeros((n,n))
#         dtau_dqd = np.zeros((n,n))


#         #backward pass
#         for i in range(n-1,-1,-1):

#             t1[:,i] = IC[i] @ S[:,i]
#             t2[:,i] = BC[i] @ S[:,i].T + IC[i] @ Sj[:,i]
#             t3[:,i] = BC[i] @ Sd[:,i] + IC[i] @ Sdd[:,i] + self.icrf(f[:,i]) @ S[:,i]
#             t4[:,i] = BC[i].T @ S[:,i]

#             subtree_ids = self.robot.get_subtree_by_id(i) #list of all subtree ids (inclusive)



#             dtau_dq[i, subtree_ids[1:]] = S[:,i] @ t3[:,subtree_ids[1:]]
            
#             dtau_dq[subtree_ids[0:], i] = Sdd[:,i] @ t1[:,subtree_ids[0:]] + \
#                                         Sd[:,i] @ t4[:,subtree_ids[0:]] 
            
#             dtau_dqd[i, subtree_ids[1:]] = S[:,i] @ t2[:,subtree_ids[1:]]

#             dtau_dqd[subtree_ids[0:], i] = Sj[:,i] @ t1[:,subtree_ids[0:]] + \
#                                         S[:,i] @ t4[:,subtree_ids[0:]] 

#             p = self.robot.get_parent_id(i)
#             if p >= 0:
#                 IC[p] = IC[p] + IC[i]
#                 BC[p] = BC[p] + BC[i]
#                 f[:,p] = f[:,p] + f[:,i]


#         return dtau_dq, dtau_dqd


#     def minv_bpass(self, q):
#         # allocate memory
#         n = len(q)
#         Minv = np.zeros((n,n))
#         F = np.zeros((n,6,n))
#         U = np.zeros((n,6))
#         Dinv = np.zeros(n)

#         # set initial IA to I
#         IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())
        
#         # backward pass
#         for ind in range(n-1,-1,-1):
#             # Compute U, D
#             S = self.robot.get_S_by_id(ind)
#             subtreeInds = self.robot.get_subtree_by_id(ind)
#             U[ind,:] = np.matmul(IA[ind],S)
#             Dinv[ind] = 1/np.matmul(S.transpose(),U[ind,:])
#             # Update Minv
#             Minv[ind,ind] = Dinv[ind]
#             for subInd in subtreeInds:
#                 Minv[ind,subInd] -= Dinv[ind] * np.matmul(S.transpose(),F[ind,:,subInd])
#             # update parent if applicable
#             parent_ind = self.robot.get_parent_id(ind)
#             if parent_ind != -1:
#                 Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#                 # update F
#                 for subInd in subtreeInds:
#                     F[ind,:,subInd] += U[ind,:]*Minv[ind,subInd]
#                     F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[ind,:,subInd]) 
#                 # update IA
#                 Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
#                 IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
#                 IA[parent_ind] += IaParent

#         return (Minv, F, U, Dinv)

#     def minv_fpass(self, q, Minv, F, U, Dinv):
#         n = len(q)
        
#         # forward pass
#         for ind in range(n):
#             parent_ind = self.robot.get_parent_id(ind)
#             S = self.robot.get_S_by_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#             if parent_ind != -1:
#                 Minv[ind,ind:] -= Dinv[ind]*np.matmul(np.matmul(U[ind,:].transpose(),Xmat),F[parent_ind,:,ind:])
            
#             F[ind,:,ind:] = np.outer(S,Minv[ind,ind:])
#             if parent_ind != -1:
#                 F[ind,:,ind:] += np.matmul(Xmat,F[parent_ind,:,ind:])

#         return Minv

#     def minv(self, q, output_dense = True):
#         # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
#         """ Computes the analytical inverse of the joint space inertia matrix
#         CRBA calculates the joint space inertia matrix H to represent the composite inertia
#         This is used in the fundamental motion equation H qdd + C = Tau
#         Forward dynamics roughly calculates acceleration as H_inv ( Tau - C); analytic inverse - benchmark against Pinocchio
#         """
#         # backward pass
#         (Minv, F, U, Dinv) = self.minv_bpass(q)

#         # forward pass
#         Minv = self.minv_fpass(q, Minv, F, U, Dinv)

#         # fill in full matrix (currently only upper triangular)
#         if output_dense:
#             n = len(q)
#             for col in range(n):
#                 for row in range(n):
#                     if col < row:
#                         Minv[row,col] = Minv[col,row]

#         return Minv

#     def crm(v):
#         if len(v) == 6:
#             vcross = np.array([0, -v[3], v[2], 0,0,0], [v[3], 0, -v[1], 0,0,0], [-v[2], v[1], 0, 0,0,0], [0, -v[6], v[5], 0,-v[3],v[2]], [v[6], 0, -v[4], v[3],0,-v[1]], [-v[5], v[4], 0, -v[2],v[1],0])
#         else:
#             vcross = np.array([0, 0, 0], [v[3], 0, -v[1]], [-v[2], v[1], 0])
#         return vcross

#     def aba(self, q, qd, tau, GRAVITY = -9.81):
#        # allocate memory
#         n = len(qd)
#         v = np.zeros((6,n))
#         c = np.zeros((6,n))
#         a = np.zeros((6,n))
#         f = np.zeros((6,n))
#         d = np.zeros(n)
#         U = np.zeros((6,n))
#         u = np.zeros(n)
#         IA = np.zeros((6,6,n))
#         pA = np.zeros((6,n))
#         qdd = np.zeros(n)
        
        
#         gravity_vec = np.zeros((6))
#         gravity_vec[5] = -GRAVITY # a_base is gravity vec
                 
#         for ind in range(n):
#             parent_ind = self.robot.get_parent_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#             S = self.robot.get_S_by_id(ind)

#             if parent_ind == -1: # parent is base
#                 v[:,ind] = S*qd[ind]
                
#             else:
#                 v[:,ind] = np.matmul(Xmat,v[:,parent_ind])
#                 v[:,ind] += S*qd[ind]
#                 c[:,ind] = self.mxS(S,v[:,ind],qd[ind])

#             Imat = self.robot.get_Imat_by_id(ind)

#             IA[:,:,ind] = Imat

#             vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
#             [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
#             [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
#             [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
#             [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
#             [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

#             crf=-np.transpose(vcross)
#             temp=np.matmul(crf,Imat)

#             pA[:,ind]=np.matmul(temp,v[:,ind])[0]
        
#         for ind in range(n-1,-1,-1):
#             S = self.robot.get_S_by_id(ind)
#             parent_ind = self.robot.get_parent_id(ind)

#             U[:,ind] = np.matmul(IA[:,:,ind],S)
#             d[ind] = np.matmul(np.transpose(S),U[:,ind])
#             u[ind] = tau[ind] - np.matmul(np.transpose(S),pA[:,ind])

#             if parent_ind != -1:

#                 rightSide=np.reshape(U[:,ind],(6,1))@np.reshape(U[:,ind],(6,1)).T/d[ind]
#                 Ia = IA[:,:,ind] - rightSide

#                 pa = pA[:,ind] + np.matmul(Ia, c[:,ind]) + U[:,ind]*u[ind]/d[ind]

#                 Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
#                 temp = np.matmul(np.transpose(Xmat), Ia)

#                 IA[:,:,parent_ind] = IA[:,:,parent_ind] + np.matmul(temp,Xmat)

#                 temp = np.matmul(np.transpose(Xmat), pa)
#                 pA[:,parent_ind]=pA[:,parent_ind] + temp.flatten()
                                             
#         for ind in range(n):

#             parent_ind = self.robot.get_parent_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

#             if parent_ind == -1: # parent is base
#                 a[:,ind] = np.matmul(Xmat,gravity_vec) + c[:,ind]
#             else:
#                 a[:,ind] = np.matmul(Xmat, a[:,parent_ind]) + c[:,ind]

#             S = self.robot.get_S_by_id(ind)
#             temp = u[ind] - np.matmul(np.transpose(U[:,ind]),a[:,ind])
#             qdd[ind] = temp / d[ind]
#             a[:,ind] = a[:,ind] + qdd[ind]*S
        
#         return qdd
    
#     def crba( self, q, qd, tau):
#         n = len(qd)
        
#         C = self.rnea(q, qd, qdd = None, GRAVITY = -9.81)[0]

#         IC = copy.deepcopy(self.robot.get_Imats_dict_by_id())# composite inertia calculation

#         for ind in range(n-1,-1,-1):
#             parent_ind = self.robot.get_parent_id(ind)
#             Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

#             if parent_ind != -1:
#                 IC[parent_ind] = IC[parent_ind] + np.matmul(np.matmul(Xmat.T, IC[ind]),Xmat);

#         H = np.zeros((n,n))

#         for ind in range(n):

#             S = self.robot.get_S_by_id(ind)
#             fh = np.matmul(IC[ind],S)
#             H[ind,ind] = np.matmul(S,fh)
#             j = ind;

#             while self.robot.get_parent_id(j) > -1:
#                 Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
#                 fh = np.matmul(Xmat.T,fh);
#                 j = self.robot.get_parent_id(j)
#                 S = self.robot.get_S_by_id(j)
#                 H[ind,j] = np.matmul(S.T, fh);
#                 H[j,ind] = H[ind,j];

#         sub=np.subtract(tau,C)
#         return H

    
