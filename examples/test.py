import numpy as np

def jacobian_tot_state(q, qd, dJdq, d2Jdq2, offsets=[np.matrix([[0,1,0,1]])]):
    """
    Compute the derivative of Jtot with respect to the state vector x = [q1, q2, q1d, q2d].
    
    Parameters:
    - q: list or array of shape (2,) with q1 and q2
    - qd: list or array of shape (2,) with q1d and q2d
    - dJdq: array of shape (4, 2) with the first derivatives of J w.r.t. q1 and q2
    - d2Jdq2: array of shape (4, 2) with the second derivatives of J w.r.t. q1 and q2
    - offsets: optional parameter for offsets (not used in this implementation)
    
    Returns:
    - dJtot_dx: array of shape (4, 4) with the derivative of Jtot w.r.t. x
    """
    # Reshape derivatives to appropriate sizes
    dJdq = dJdq.reshape(2, 2, 2)
    d2Jdq2 = d2Jdq2.reshape(2, 2, 2, 2)
    
    # Compute J and J2
    J = np.array([[q[0], 0], [0, q[1]]])  # Placeholder for J(q)
    dJ_dq = dJdq
    J2 = (dJ_dq @ qd).reshape(2, 2)
    
    # Initialize the derivative matrix dJtot_dx
    dJtot_dx = np.zeros((4, 4))
    
    # Compute the partial derivatives
    dJtot_dq1 = np.zeros((4, 4))
    dJtot_dq2 = np.zeros((4, 4))
    dJtot_dq1d = np.zeros((4, 4))
    dJtot_dq2d = np.zeros((4, 4))
    
    # Top left block is the derivative of J with respect to q1 and q2
    dJtot_dq1[0:2, 0:2] = dJ_dq[:, :, 0]
    dJtot_dq2[0:2, 0:2] = dJ_dq[:, :, 1]
    
    # Bottom left block is the derivative of J2 with respect to q1 and q2
    dJtot_dq1[2:4, 0:2] = (d2Jdq2[:, :, 0, 0] * qd[0] + d2Jdq2[:, :, 0, 1] * qd[1]).reshape(2, 2)
    dJtot_dq2[2:4, 0:2] = (d2Jdq2[:, :, 1, 0] * qd[0] + d2Jdq2[:, :, 1, 1] * qd[1]).reshape(2, 2)
    
    # Bottom right block is the derivative of J with respect to q1 and q2 (copied from top left)
    dJtot_dq1[2:4, 2:4] = dJ_dq[:, :, 0]
    dJtot_dq2[2:4, 2:4] = dJ_dq[:, :, 1]
    
    # Bottom left block also has contributions from the derivative with respect to q1d and q2d
    dJtot_dq1d[2:4, 0:2] = dJ_dq[:, :, 0]
    dJtot_dq2d[2:4, 0:2] = dJ_dq[:, :, 1]
    
    # Sum the contributions to form the total derivative
    dJtot_dx = dJtot_dq1 + dJtot_dq2 + dJtot_dq1d + dJtot_dq2d
    
    return dJtot_dx

# Example usage:
q = [1, 1]
qd = [1, 2]
dJdq = np.random.rand(4, 2)  # Example values
d2Jdq2 = np.random.rand(4, 2)  # Example values

dJtot_dx = jacobian_tot_state(q, qd, dJdq, d2Jdq2)
print(dJtot_dx)
