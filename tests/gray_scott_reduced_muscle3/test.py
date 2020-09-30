#I had to manually install ruamel.yaml to get these import to work
from libmuscle import Grid, Instance, Message
from ymmsl import ComputeElement, Conduit, Configuration, Model, Operator, Settings

import numpy as np
import easysurrogate as es


def reduced_sgs():
    """    
    """
    
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
            reduced_dict_u = surrogate.train([V_hat_1, u_hat], Q_ref[0:N_Q], Q_model[0:N_Q])
            reduced_dict_v = surrogate.train([V_hat_1, v_hat], Q_ref[N_Q:], Q_model[N_Q:])
        
            # get the two reduced sgs terms from the dict
            reduced_sgs_u = np.fft.ifft2(reduced_dict_u['sgs_hat'])
            reduced_sgs_v = np.fft.ifft2(reduced_dict_v['sgs_hat'])
            
        #O_F
        instance.send('reduced_sgs_u', Message(t_cur, None, Grid(reduced_sgs_u)))
        instance.send('reduced_sgs_v', Message(t_cur, None, Grid(reduced_sgs_v)))

        