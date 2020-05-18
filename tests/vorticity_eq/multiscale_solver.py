"""
*************************
* S U B R O U T I N E S *
*************************
"""

#pseudo-spectral technique to solve for Fourier coefs of Jacobian
def compute_VgradW_hat(w_hat_n, P):
    
    #compute streamfunction
    psi_hat_n = w_hat_n/k_squared_no_zero
    psi_hat_n[0,0] = 0.0
    
    #compute jacobian in physical space
    u_n = np.fft.irfft2(-ky*psi_hat_n)
    w_x_n = np.fft.irfft2(kx*w_hat_n)

    v_n = np.fft.irfft2(kx*psi_hat_n)
    w_y_n = np.fft.irfft2(ky*w_hat_n)
    
    VgradW_n = u_n*w_x_n + v_n*w_y_n
    
    #return to spectral space
    VgradW_hat_n = np.fft.rfft2(VgradW_n)
    
    VgradW_hat_n *= P
    
    return VgradW_hat_n

#get Fourier coefficient of the vorticity at next (n+1) time step
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor, sgs_hat = 0.0):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat - sgs_hat)
    
    return w_hat_np1, VgradW_hat_n

def draw():
    
    ax1.contourf(x, y, Q1, 50)
    ax2.contourf(x, y, Q2, 50)
    plt.pause(0.1)
    plt.tight_layout()
    
#compute spectral filter
def get_P(cutoff):
    
    P = np.ones([N, int(N/2+1)])
    
    for i in range(N):
        for j in range(int(N/2+1)):
            
            if np.abs(kx[i, j]) > cutoff or np.abs(ky[i, j]) > cutoff:
                P[i, j] = 0.0
                
    return P

#store samples in hierarchical data format, when sample size become very large
def store_samples_hdf5():
  
    fname = HOME + '/samples/' + store_ID + '_t_' + str(np.around(t_end/day, 1)) + '.hdf5'
    
    print('Storing samples in ', fname)
    
    if os.path.exists(HOME + '/samples') == False:
        os.makedirs(HOME + '/samples')
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for q in QoI:
        h5f.create_dataset(q, data = samples[q])
        
    h5f.close()  

#compute the spatial correlation coeffient at a given time
def spatial_corr_coef(X, Y):
    return np.mean((X - np.mean(X))*(Y - np.mean(Y)))/(np.std(X)*np.std(Y))

"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import matplotlib.pyplot as plt
import os, h5py

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

HOME = os.path.abspath(os.path.dirname(__file__))

#number of gridpoints in 1D
I = 6
N = 2**I            #HF (High-fidelity or reference grid)
N_LF = 2**(I-1)     #LF (Low-Fidelity or resolved grid)

#2D grid
h = 2*np.pi/N
axis = h*np.arange(1, N+1)
axis = np.linspace(0, 2.0*np.pi, N)
[x , y] = np.meshgrid(axis , axis)

#frequencies
k = np.fft.fftfreq(N)*N

kx = np.zeros([N, int(N/2+1)]) + 0.0j
ky = np.zeros([N, int(N/2+1)]) + 0.0j

for i in range(N):
    for j in range(int(N/2+1)):
        kx[i, j] = 1j*k[j]
        ky[i, j] = 1j*k[i]

k_squared = kx**2 + ky**2
k_squared_no_zero = np.copy(k_squared)
k_squared_no_zero[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = N_LF/3

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
nu_LF = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time (in days) + time step
t_burn = 365*day
t = 0*day
t_end = t + 5*365*day
dt = 0.01

n_steps = np.ceil((t_end-t)/dt).astype('int')
n_burn = np.ceil((t_burn-t)/dt).astype('int')

#constant factor that appears in AB/BDI2 time stepping scheme, multiplying the Fourier coefficient w_hat_np1
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)
norm_factor_LF = 1.0/(3.0/(2.0*dt) - nu_LF*k_squared + mu)

#############
# USER KEYS #
#############

#framerate of storing data, plotting results (1 = every integration time step)
store_frame_rate = np.floor(0.25*day/dt).astype('int')
plot_frame_rate = np.floor(1*day/dt).astype('int')

#length of data array
S = np.ceil((n_steps-n_burn)/store_frame_rate).astype('int')

sim_ID = 'run1'
#store the state at the end of the simulation
state_store = True
#restart from a stored state
restart = False
#store data
store = True
store_ID = sim_ID 
#plot while running
plot = False

###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#TRAINING DATA SET
QoI = ['w_hat_nm1_LF', 'VgradW_hat_nm1_LF', 'r_hat_nm1']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    
    for q in range(Q):
        
        #assume a field contains the string '_hat_'
        if '_hat' in QoI[q]:
            #store Fourier coefs
            samples[QoI[q]] = np.zeros([S, N, int(N/2+1)]) + 0.0j
            #store full fields
            # samples[QoI[q]] = np.zeros([S, N, N]) 
        #a scalar
        else:
            samples[QoI[q]] = np.zeros(S)

#forcing term
F = 2**1.5*np.cos(5*x)*np.cos(5*y);
F_hat = np.fft.rfft2(F);

#restart from a previous stored state (set restart = True, and set t to the end time of previous simulation)
if restart == True:
    
    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t/day,1)) + '.hdf5'
    
    #create HDF5 file
    h5f = h5py.File(fname, 'r')
    
    for key in h5f.keys():
        print(key)
        vars()[key] = h5f[key][:]
        
    h5f.close()
#start from initial condition
else:
    
    #initial condition
    w = np.sin(4.0*x)*np.sin(4.0*y) + 0.4*np.cos(3.0*x)*np.cos(3.0*y) + \
        0.3*np.cos(5.0*x)*np.cos(5.0*y) + 0.02*np.sin(x) + 0.02*np.cos(y)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_HF = P*np.fft.rfft2(w)
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
   
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_HF = compute_VgradW_hat(w_hat_n_HF, P)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    #initial Fourier coefficients at time n and n-1
    w_hat_n_LF = P*np.fft.rfft2(w)
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
   
    #initial Fourier coefficients of the jacobian at time n and n-1
    VgradW_hat_n_LF = compute_VgradW_hat(w_hat_n_LF, P_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

    t = 0.0


print('Solving forced dissipative vorticity equations')
print('Ref Grid = ', N, 'x', N)
print('Grid = ', N_LF, 'x', N_LF)
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')

j1 = 0; j2 = 0; idx = 0

if plot:
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

#time loop
for n in range(n_steps):
    
    #HF solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)

    #subgrid scale term
    r_hat_nm1 = P_LF*VgradW_hat_nm1_HF - VgradW_hat_nm1_LF

    #Euler approximation of the covariate nabla^2(Dw/DT)
    # covar_zanna_hat_n = P_LF*k_squared*((w_hat_n_LF - w_hat_nm1_LF)/dt + VgradW_hat_nm1_LF)

    #LF solve for next time step
    w_hat_np1_LF, VgradW_hat_n_LF = get_w_hat_np1(w_hat_n_LF, w_hat_nm1_LF, VgradW_hat_nm1_LF, P_LF, norm_factor_LF, sgs_hat = r_hat_nm1)

    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0

        if t > t_burn:
            for qoi in QoI:
                #store Fourier coefs
                samples[qoi][idx] = eval(qoi)
                #store full fields
                # samples[qoi][idx] = np.fft.irfft2(eval(qoi))

            idx += 1
        
    if j1 == plot_frame_rate and plot == True:
        
        j1 = 0
        Q1 = np.fft.irfft2(P_LF*w_hat_n_HF)
        Q2 = np.fft.irfft2(w_hat_n_LF)
        draw()

    #update variables
    t += dt; j2 += 1; j1 += 1

    #update HF vars
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    w_hat_n_HF = np.copy(w_hat_np1_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)

    #update LF vars
    w_hat_nm1_LF = np.copy(w_hat_n_LF)
    w_hat_n_LF = np.copy(w_hat_np1_LF)
    VgradW_hat_nm1_LF = np.copy(VgradW_hat_n_LF)

    if np.mod(n, np.round(day/dt)) == 0:
        print('n = ', n, 'of', n_steps)
    
#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:

    keys = ['w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF', \
            'w_hat_nm1_LF', 'w_hat_n_LF', 'VgradW_hat_nm1_LF']

    fname = HOME + '/restart/' + sim_ID + '_t_' + str(np.around(t_end/day,1)) + '.hdf5'
    
    if os.path.exists(HOME + '/restart') == False:
        os.makedirs(HOME + '/restart')
    
    #create HDF5 file
    h5f = h5py.File(fname, 'w')
    
    #store numpy sample arrays as individual datasets in the hdf5 file
    for key in keys:
        qoi = eval(key)
        h5f.create_dataset(key, data = qoi)
        
    h5f.close()
    
#store the samples
if store == True:
    store_samples_hdf5() 
    
if plot == False:
    #plot vorticity field
    fig = plt.figure(figsize=[8, 4])
    ax = fig.add_subplot(121, xlabel=r'x', ylabel=r'y', title='t = ' + str(np.around(t/day, 2)) + ' days')
    ct = ax.contourf(x, y, np.fft.irfft2(w_hat_np1_HF), 100)
    plt.colorbar(ct)
    
    ax = fig.add_subplot(122, xlabel=r'x', ylabel=r'y', title='t = ' + str(np.around(t/day, 2)) + ' days')
    ct = ax.contourf(x, y, np.fft.irfft2(P_LF*w_hat_np1_LF), 100)
    plt.colorbar(ct)
    
    plt.tight_layout()
    
    plt.show()