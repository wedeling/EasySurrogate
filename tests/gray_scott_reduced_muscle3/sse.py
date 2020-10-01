#I had to manually install ruamel.yaml to get these import to work
from libmuscle import Instance, Message
from ymmsl import Operator

import numpy as np

###########################
# REDUCED SGS SUBROUTINES #
###########################

def reduced_r(V_hat, dQ, N_Q, N_LF):
    """
    Compute the reduced SGS term
    """
    
    #compute the T_ij basis functions
    T_hat = np.zeros([N_Q, N_Q, N_LF, N_LF]) + 0.0j
    
    for i in range(N_Q):

        T_hat[i, 0] = V_hat[i]
        
        J = np.delete(np.arange(N_Q), i)
        
        idx = 1
        for j in J:
            T_hat[i, idx] = V_hat[j]
            idx += 1

    #compute the coefficients c_ij
    inner_prods = inner_products(V_hat, N_Q, N_LF)

    c_ij = compute_cij_using_V_hat(V_hat, inner_prods, N_Q)

    EF_hat = 0.0

    src_Q = np.zeros(N_Q)
    tau = np.zeros(N_Q)

    #loop over all QoI
    for i in range(N_Q):
        #compute the fourier coefs of the P_i
        P_hat_i = T_hat[i, 0]
        for j in range(0, N_Q-1):
            P_hat_i -= c_ij[i, j]*T_hat[i, j+1]
    
        #(V_i, P_i) integral
        src_Q_i = compute_int(V_hat[i], P_hat_i, N_LF)
        
        #compute tau_i = Delta Q_i/ (V_i, P_i)
        tau_i = dQ[i]/src_Q_i        

        src_Q[i] = src_Q_i
        tau[i] = tau_i

        #compute reduced soure term
        EF_hat -= tau_i*P_hat_i
    
    return EF_hat

def compute_cij_using_V_hat(V_hat, inner_prods, N_Q):
    """
    compute the coefficients c_ij of P_i = T_{i,1} - c_{i,2}*T_{i,2}, - ...
    """

    c_ij = np.zeros([N_Q, N_Q-1])
    
    for i in range(N_Q):
        A = np.zeros([N_Q-1, N_Q-1])
        b = np.zeros(N_Q-1)

        k = np.delete(np.arange(N_Q), i)

        for j1 in range(N_Q-1):
            for j2 in range(j1, N_Q-1):
                A[j1, j2] = inner_prods[k[j1], k[j2]]
                if j1 != j2:
                    A[j2, j1] = A[j1, j2]

        for j1 in range(N_Q-1):
            b[j1] = inner_prods[i, k[j1]]

        if N_Q == 2:
            c_ij[i,:] = b/A
        else:
            c_ij[i,:] = np.linalg.solve(A, b)
            
    return c_ij

def inner_products(V_hat, N_Q,  N_LF):

    """
    Compute all the inner products (V_i, T_{i,j})
    """

    V_hat = V_hat.reshape([N_Q, N_LF**2])

    return np.dot(V_hat, np.conjugate(V_hat).T)/N_LF**4

def compute_int(X1_hat, X2_hat, N):
    """
    Compute the integral of X1*X2 using the Fourier expansion
    """
    integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten()))/N**4 
    return integral.real


def reduced_sgs():
    """    
    """
    
    N_LF = 2**7
    N_Q = 2
    
    instance = Instance({
        Operator.F_INIT: ['state'],     #a dict with state values  
        Operator.O_F: ['sgs']})         #a dict with subgrid-scale terms
    
    while instance.reuse_instance():
        
        #F_INIT
        t_max = instance.get_setting('t_max')
        dt = instance.get_setting('dt')
        N_Q = instance.get_setting('N_Q')

        msg = instance.receive('state')
        V_hat_1 = msg['V_hat_1']
        u_hat = msg['u_hat']
        v_hat = msg['v_hat']
        Q_ref = msg['Q_ref']
        Q_model = msg['Q_model']

        t_cur = msg.timestamp
        
        while t_cur + dt <= msg.timestamp + t_max:

            #S

            # train the two reduced sgs source terms
            sgs_hat_u = reduced_sgs([V_hat_1, u_hat], Q_ref[0:N_Q] - Q_model[0:N_Q], N_Q, N_LF)
            sgs_hat_v = reduced_sgs([V_hat_1, v_hat], Q_ref[N_Q:] - Q_model[N_Q:], N_Q, N_LF)
        
            # get the two reduced sgs terms from the dict
            reduced_sgs_u = np.fft.ifft2(sgs_hat_u)
            reduced_sgs_v = np.fft.ifft2(sgs_hat_v)
            
        #O_F
        instance.send('sgs', Message(t_cur, None, 
                                     {'reduced_sgs_u':reduced_sgs_u, 'reduced_sgs_v':reduced_sgs_v}))

if __name__ == '__main__':
    reduced_sgs()