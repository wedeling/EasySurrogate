"""
A 2D Gray-Scott reaction diffusion model.

Numerical method:
Craster & Sassi, spectral algorithmns for reaction-diffusion equations, 2006.

This script is used to generate the training data for the reduced
subgrid-scale model
"""

from libmuscle import Instance, Message, Grid
from ymmsl import Operator

import numpy as np
import time
import h5py
import tkinter as tk
from tkinter import filedialog
import os

def get_grid(N, L):
    """
    Generate an equidistant N x N square grid

    Parameters
    ----------
    N : number of point in 1 dimension

    Returns
    -------
    xx, yy: the N x N coordinates

    """
    x = (2 * L / N) * np.arange(-N / 2, N / 2)
    y = x
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def get_derivative_operator(N, L):
    """
    Get the spectral operators used to compute the spatial dervatives in
    x and y direction

    Parameters
    ----------
    N : number of points in 1 dimension

    Returns
    -------
    kx, ky: operators to compute derivatives in spectral space. Already
    multiplied by the imaginary unit 1j

    """
    # frequencies of fft2
    k = np.fft.fftfreq(N) * N
    # frequencies must be scaled as well
    k = k * np.pi / L
    kx = np.zeros([N, N]) + 0.0j
    ky = np.zeros([N, N]) + 0.0j

    for i in range(N):
        for j in range(N):
            kx[i, j] = 1j * k[j]
            ky[i, j] = 1j * k[i]

    return kx, ky


def get_spectral_filter(kx, ky, N, cutoff):
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):

            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0

    return P


def initial_cond(xx, yy):
    """
    Compute the initial condition

    Parameters
    ----------
    xx : spatial grid points in x direction
    yy : spatial grid points in y direction

    Returns
    -------
    u_hat, v_hat: initial Fourier coefficients of u and v

    """
    common_exp = np.exp(-10 * (xx**2 / 2 + yy**2)) + \
        np.exp(-50 * ((xx - 0.5)**2 + (yy - 0.5)**2))
    u = 1 - 0.5 * common_exp
    v = 0.25 * common_exp
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    return u_hat, v_hat


def integrating_factors(k_squared, dt, epsilon_u, epsilon_v):
    """
    Compute the integrating factors used in the RK4 time stepping

    Parameters
    ----------
    k_squared : the operator to compute the Laplace operator

    Returns
    -------
    The integrating factors for u and v

    """

    int_fac_u = np.exp(epsilon_u * k_squared * dt / 2)
    int_fac_u2 = np.exp(epsilon_u * k_squared * dt)
    int_fac_v = np.exp(epsilon_v * k_squared * dt / 2)
    int_fac_v2 = np.exp(epsilon_v * k_squared * dt)

    return int_fac_u, int_fac_u2, int_fac_v, int_fac_v2


def rhs_hat(u_hat, v_hat, feed, kill, **kwargs):
    """
    Right hand side of the 2D Gray-Scott equations

    Parameters
    ----------
    u_hat : Fourier coefficients of u
    v_hat : Fourier coefficients of v

    Returns
    -------
    The Fourier coefficients of the right-hand side of u and v (f_hat & g_hat)

    """

    #######################
    # MUSCLE modification #
    #######################

    if 'reduced_sgs_u' in kwargs and 'reduced_sgs_v' in kwargs:
        reduced_sgs_u = kwargs['reduced_sgs_u']
        reduced_sgs_v = kwargs['reduced_sgs_v']
    else:
        reduced_sgs_u = reduced_sgs_v = 0

    u = np.fft.ifft2(u_hat)
    v = np.fft.ifft2(v_hat)

    f = -u * v * v + feed * (1 - u) - reduced_sgs_u
    g = u * v * v - (feed + kill) * v - reduced_sgs_v

    ###########################
    # End MUSCLE modification #
    #######################

    f_hat = np.fft.fft2(f)
    g_hat = np.fft.fft2(g)

    return f_hat, g_hat


def rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2, dt, feed, kill, **kwargs):
    """
    Runge-Kutta 4 time-stepping subroutine

    Parameters
    ----------
    u_hat : Fourier coefficients of u
    v_hat : Fourier coefficients of v

    Returns
    -------
    u_hat and v_hat at the next time step

    """
    # RK4 step 1
    k_hat_1, l_hat_1 = rhs_hat(u_hat, v_hat, feed, kill, **kwargs)
    k_hat_1 *= dt
    l_hat_1 *= dt
    u_hat_2 = (u_hat + k_hat_1 / 2) * int_fac_u
    v_hat_2 = (v_hat + l_hat_1 / 2) * int_fac_v
    # RK4 step 2
    k_hat_2, l_hat_2 = rhs_hat(u_hat_2, v_hat_2, feed, kill, **kwargs)
    k_hat_2 *= dt
    l_hat_2 *= dt
    u_hat_3 = u_hat * int_fac_u + k_hat_2 / 2
    v_hat_3 = v_hat * int_fac_v + l_hat_2 / 2
    # RK4 step 3
    k_hat_3, l_hat_3 = rhs_hat(u_hat_3, v_hat_3, feed, kill, **kwargs)
    k_hat_3 *= dt
    l_hat_3 *= dt
    u_hat_4 = u_hat * int_fac_u2 + k_hat_3 * int_fac_u
    v_hat_4 = v_hat * int_fac_v2 + l_hat_3 * int_fac_v
    # RK4 step 4
    k_hat_4, l_hat_4 = rhs_hat(u_hat_4, v_hat_4, feed, kill, **kwargs)
    k_hat_4 *= dt
    l_hat_4 *= dt
    u_hat = u_hat * int_fac_u2 + 1 / 6 * (k_hat_1 * int_fac_u2 +
                                          2 * k_hat_2 * int_fac_u +
                                          2 * k_hat_3 * int_fac_u +
                                          k_hat_4)
    v_hat = v_hat * int_fac_v2 + 1 / 6 * (l_hat_1 * int_fac_v2 +
                                          2 * l_hat_2 * int_fac_v +
                                          2 * l_hat_3 * int_fac_v +
                                          l_hat_4)
    return u_hat, v_hat

def compute_int(X1_hat, X2_hat, N):
    """
    Compute the integral of X1*X2 using the Fourier expansion
    """
    integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten())) / N**4
    return integral.real


def gray_scott_macro():
    #######################
    # MUSCLE modification #
    #######################
    
    #define the MUSCLE in and out ports
    instance = Instance({
        Operator.O_I: ['state_out'],
        Operator.S: ['state_in']})
    
    while instance.reuse_instance():
        # plt.close('all')
        # plt.rcParams['image.cmap'] = 'seismic'
        HOME = os.path.abspath(os.path.dirname(__file__))
        
        #load the reference data for EasySurrogate
        # fname = os.path.join(HOME, 'samples/gray_scott_reference.hdf5')
        # h5f = h5py.File(fname, 'r')
        # ref_data = h5f['Q_HF'][()]
       
        # alpha pattern
        feed = instance.get_setting('feed')
        kill = instance.get_setting('kill')
     
        ###########################
        # End MUSCLE modification #
        ###########################
     
        # number of gridpoints in 1D
        I = 7
        N = 2**I
        N_ref = 2**(I+1)
        
        # number of time series to track
        N_Q = 2
        
        # domain size [-L, L]
        L = 1.25
        
        # user flags
        store = True
        state_store = True
        restart = False
        
        sim_ID = 'test_gray_scott'
        
        # TRAINING DATA SET
        QoI = ['Q_HF', 'Q_ref']
        Q = len(QoI)
        
        # allocate memory
        samples = {}
        
        if store:
            samples['N'] = N
        
            for q in range(Q):
                samples[QoI[q]] = []
        
        # 2D grid, scaled by L
        xx, yy = get_grid(N, L)
        xx_ref, yy_ref = get_grid(N_ref, L)
       
        # spatial derivative operators
        kx, ky = get_derivative_operator(N, L)
        kx_ref, ky_ref = get_derivative_operator(N_ref, L)

        # Laplace operator
        k_squared = kx**2 + ky**2
        k_squared_ref = kx_ref**2 + ky_ref**2

        # diffusion coefficients
        epsilon_u = 2e-5
        epsilon_v = 1e-5
        
        # time step parameters
        dt = 0.1
        n_steps = 100
        store_frame_rate = 1
        t = 0.0
        
        # Initial condition
        if restart:
        
            fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t, 1)) + '.hdf5'
        
            # if fname does not exist, select restart file via GUI
            if os.path.exists(fname) == False:
                root = tk.Tk()
                root.withdraw()
                fname = filedialog.askopenfilename(initialdir=HOME + '/restart',
                                                   title="Open restart file",
                                                   filetypes=(('HDF5 files', '*.hdf5'),
                                                              ('All files', '*.*')))
        
            # create HDF5 file
            h5f = h5py.File(fname, 'r')
        
            for key in h5f.keys():
                print(key)
                vars()[key] = h5f[key][:]
        
            h5f.close()
        else:
            u_hat, v_hat = initial_cond(xx, yy)
            u_hat_ref, v_hat_ref = initial_cond(xx_ref, yy_ref)

        # Integrating factors
        int_fac_u, int_fac_u2, int_fac_v, int_fac_v2 = \
            integrating_factors(k_squared, dt, epsilon_u, epsilon_v)

        int_fac_u_ref, int_fac_u2_ref, int_fac_v_ref, int_fac_v2_ref = \
            integrating_factors(k_squared_ref, dt, epsilon_u, epsilon_v)
       
        # counters
        j = 0
        j2 = 0
        
        V_hat_1 = np.fft.fft2(np.ones([N, N]))
        V_hat_1_ref = np.fft.fft2(np.ones([N_ref, N_ref]))

        t0 = time.time()
        
        samples_uq = np.zeros([n_steps, 8])

        # time stepping
        for n in range(n_steps):

            # compute reference stats
            Q_HF = np.zeros(2 * N_Q)
            Q_HF[0] = compute_int(V_hat_1, u_hat, N)
            Q_HF[1] = 0.5 * compute_int(u_hat, u_hat, N)
            Q_HF[2] = compute_int(V_hat_1, v_hat, N)
            Q_HF[3] = 0.5 * compute_int(v_hat, v_hat, N)

            Q_ref = np.zeros(2 * N_Q)
            Q_ref[0] = compute_int(V_hat_1_ref, u_hat_ref, N_ref)
            Q_ref[1] = 0.5 * compute_int(u_hat_ref, u_hat_ref, N_ref)
            Q_ref[2] = compute_int(V_hat_1_ref, v_hat_ref, N_ref)
            Q_ref[3] = 0.5 * compute_int(v_hat_ref, v_hat_ref, N_ref)

            samples_uq[n, 0:2 * N_Q] = Q_HF
            samples_uq[n, 2 * N_Q:] = Q_ref

            if np.mod(n, 100) == 0:
                print('time step %d of %d' % (n, n_steps))
                print(Q_HF)
                print(Q_ref)

            #######################
            # MUSCLE modification #
            #######################

            #MUSCLE O_I port (state_out)
            t_cur = n*dt
            t_next = t_cur + dt
            if n == n_steps - 1:
                t_next = None
            
            #split state vars in real and imag part (temporary fix)
            V_hat_1_re = np.copy(V_hat_1.real)
            V_hat_1_im = np.copy(V_hat_1.imag)
            u_hat_re = np.copy(u_hat.real)
            u_hat_im = np.copy(u_hat.imag)
            v_hat_re = np.copy(v_hat.real)
            v_hat_im = np.copy(v_hat.imag)
                        
            #create a MUSCLE Message object, to be sent to the micro model
            cur_state = Message(t_cur, t_next, {'V_hat_1_re': V_hat_1_re, 
                                                'V_hat_1_im': V_hat_1_im, 
                                                'u_hat_re': u_hat_re,
                                                'u_hat_im': u_hat_im,                                                
                                                'v_hat_re': v_hat_re,
                                                'v_hat_im': v_hat_im,                                               
                                                'Q_ref': Q_ref, 'Q_model': Q_HF})
            #send the message to the micro model
            instance.send('state_out', cur_state)
    
            #reveive a message from the micro model, i.e. the two reduced subgrid-scale terms
            msg = instance.receive('state_in')
            reduced_sgs_u_re = msg.data['reduced_sgs_u_re'].array
            reduced_sgs_u_im = msg.data['reduced_sgs_u_im'].array
            reduced_sgs_v_re = msg.data['reduced_sgs_v_re'].array
            reduced_sgs_v_im = msg.data['reduced_sgs_v_im'].array
            
            #recreate the reduced subgrid-scale terms for the u and v pde
            reduced_sgs_u = reduced_sgs_u_re + 1.0j*reduced_sgs_u_im
            reduced_sgs_v = reduced_sgs_v_re + 1.0j*reduced_sgs_v_im

            ###########################
            # End MUSCLE modification #
            ###########################

            if np.mod(n, 1000) == 0:
                print('time step %d of %d' % (n, n_steps))

            #evolve the state in time, with the new reduced sgs terms
            u_hat, v_hat = rk4(u_hat, v_hat, int_fac_u, int_fac_u2, int_fac_v, int_fac_v2, dt, feed, kill,
                               reduced_sgs_u=reduced_sgs_u, reduced_sgs_v=reduced_sgs_v)

            u_hat_ref, v_hat_ref = rk4(u_hat_ref, v_hat_ref, int_fac_u_ref, int_fac_u2_ref, 
                                       int_fac_v_ref, int_fac_v2_ref, dt, feed, kill)

            j += 1
            j2 += 1
            t += dt

            if j2 == store_frame_rate and store:
                j2 = 0

                for qoi in QoI:
                    samples[qoi].append(eval(qoi))

    t1 = time.time()
    print('*************************************')
    print('Simulation time = %f [s]' % (t1 - t0))
    print('*************************************')

    # output csv file
    header = 'Q1,Q2,Q3,Q4,Q1_HF,Q2_HF,Q3_HF,Q4_HF'
    fname = 'output_f%.3f_k%.3f.csv' % (feed, kill)
    np.savetxt(fname, samples_uq,
               delimiter=",", comments='',
               header=header)
    
    # store the state of the system to allow for a simulation restart at t > 0
    if state_store:
    
        keys = ['u_hat', 'v_hat']
    
        if os.path.exists(HOME + '/restart') == False:
            os.makedirs(HOME + '/restart')
    
        fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t, 1)) + '.hdf5'
    
        # create HDF5 file
        h5f = h5py.File(fname, 'w')
    
        # store numpy sample arrays as individual datasets in the hdf5 file
        for key in keys:
            qoi = eval(key)
            h5f.create_dataset(key, data=qoi)
    
        h5f.close()
    
if __name__ == '__main__':
    gray_scott_macro()