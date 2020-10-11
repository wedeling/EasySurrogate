from libmuscle import Instance, Message
from ymmsl import Operator

import numpy as np
import logging
import easysurrogate as es

def reduced_sgs():
    """
    An EasySurrogate Reduced micro model, executed in a separate file and linked to the
    macro model via MUSCLE3
    """
   
    instance = Instance({
        Operator.F_INIT: ['state'],     #a dict with state values  
        Operator.O_F: ['sgs']})         #a dict with subgrid-scale terms

    #get some parameters
    N_Q = instance.get_setting('N_Q')       #the number of QoI to track, per PDE
    N_LF = instance.get_setting('N_LF')     #the number of gridpoints in 1 dimension
    t_max = instance.get_setting('t_max')   #the simulation time, per time-step of macro   
    dt = instance.get_setting('dt')         #the micro time step

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    #Create an EasySurrogate RecucedSurrogate object
    surrogate = es.methods.Reduced_Surrogate(N_Q, N_LF)

    while instance.reuse_instance():

        #receive the state from the macro model
        msg = instance.receive('state')
        V_hat_1_re = msg.data['V_hat_1_re'].array.T
        V_hat_1_im = msg.data['V_hat_1_im'].array.T
        u_hat_re = msg.data['u_hat_re'].array.T
        u_hat_im = msg.data['u_hat_im'].array.T
        v_hat_re = msg.data['v_hat_re'].array.T
        v_hat_im = msg.data['v_hat_im'].array.T
        Q_ref = msg.data['Q_ref'].array.T
        Q_model = msg.data['Q_model'].array.T

        #recreate the Fourier coefficients (temporary fix)
        V_hat_1 = V_hat_1_re + 1.0j*V_hat_1_im
        u_hat = u_hat_re + 1.0j*u_hat_im
        v_hat = v_hat_re + 1.0j*v_hat_im

        # #time of the macro model
        t_cur = msg.timestamp
        
        # train the two reduced sgs source terms using the recieved reference data Q_ref
        reduced_dict_u = surrogate.train([V_hat_1, u_hat], Q_ref[0:N_Q], Q_model[0:N_Q])
        reduced_dict_v = surrogate.train([V_hat_1, v_hat], Q_ref[N_Q:], Q_model[N_Q:])

        # get the two reduced sgs terms from the dict
        reduced_sgs_u = np.fft.ifft2(reduced_dict_u['sgs_hat'])
        reduced_sgs_v = np.fft.ifft2(reduced_dict_v['sgs_hat'])

        #MUSCLE O_F port (sgs), send the subgrid-scale terms back to the macro model
        reduced_sgs_u_re = np.copy(reduced_sgs_u.real)
        reduced_sgs_u_im = np.copy(reduced_sgs_u.imag)
        reduced_sgs_v_re = np.copy(reduced_sgs_v.real)
        reduced_sgs_v_im = np.copy(reduced_sgs_v.imag)

        instance.send('sgs', Message(t_cur, None, 
                                     {'reduced_sgs_u_re': reduced_sgs_u_re, 
                                      'reduced_sgs_u_im': reduced_sgs_u_im,
                                      'reduced_sgs_v_re': reduced_sgs_v_re,
                                      'reduced_sgs_v_im': reduced_sgs_v_im}))

if __name__ == '__main__':
    reduced_sgs()