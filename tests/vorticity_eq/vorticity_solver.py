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
def get_w_hat_np1(w_hat_n, w_hat_nm1, VgradW_hat_nm1, P, norm_factor):
    
    #compute jacobian
    VgradW_hat_n = compute_VgradW_hat(w_hat_n, P)
    
    #solve for next time step according to AB/BDI2 scheme
    w_hat_np1 = norm_factor*P*(2.0/dt*w_hat_n - 1.0/(2.0*dt)*w_hat_nm1 - \
                               2.0*VgradW_hat_n + VgradW_hat_nm1 + mu*F_hat)
    
    return w_hat_np1, VgradW_hat_n

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
I = 7
N = 2**I

#2D grid
axis = np.linspace(-np.pi, +np.pi, N)
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

#spectral filter
P = get_P(Ncutoff)

#time scale
Omega = 7.292*10**-5
day = 24*60**2*Omega

#viscosities
decay_time_nu = 5.0
decay_time_mu = 90.0
nu = 1.0/(day*Ncutoff**2*decay_time_nu)
mu = 1.0/(day*decay_time_mu)

#start, end time (in days) + time step
t_burn = 365*day
t = 0*day
t_end = 4*365*day + t_burn 
dt = 0.01

n_steps = np.ceil((t_end-t)/dt).astype('int')
n_burn = np.ceil((t_burn-t)/dt).astype('int')

#constant factor that appears in AB/BDI2 time stepping scheme, multiplying the Fourier coefficient w_hat_np1
norm_factor = 1.0/(3.0/(2.0*dt) - nu*k_squared + mu)

#############
# USER KEYS #
#############

#framerate of storing data, plotting results (1 = every integration time step)
store_frame_rate = np.floor(1.0*day/dt).astype('int')
#length of data array
S = np.floor((n_steps-n_burn)/store_frame_rate).astype('int')

sim_ID = 'run1'
#store the state at the end of the simulation
state_store = True
#restart from a stored state
restart = False 
#store data
store = True
store_ID = sim_ID

###############################
# SPECIFY WHICH DATA TO STORE #
###############################

#TRAINING DATA SET
QoI = ['w_hat_n_HF', 'VgradW_hat_nm1_HF']
Q = len(QoI)

#allocate memory
samples = {}

if store == True:
    samples['S'] = S
    samples['N'] = N
    
    for q in range(Q):
        
        #assume a field contains the string '_hat_'
        if '_hat_' in QoI[q]:
            samples[QoI[q]] = np.zeros([S, N, int(N/2+1)]) + 0.0j
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

print('Solving forced dissipative vorticity equations')
print('Grid = ', N, 'x', N)
print('t_begin = ', t/day, 'days')
print('t_end = ', t_end/day, 'days')

j2 = 0; idx = 0

#time loop
for n in range(n_steps):
    
    #solve for next time step
    w_hat_np1_HF, VgradW_hat_n_HF = get_w_hat_np1(w_hat_n_HF, w_hat_nm1_HF, VgradW_hat_nm1_HF, P, norm_factor)

    #store samples to dict
    if j2 == store_frame_rate and store == True:
        j2 = 0

        if t > t_burn:
            for qoi in QoI:
                samples[qoi][idx] = eval(qoi)

            idx += 1  

    #update variables
    t += dt; j2 += 1
    w_hat_nm1_HF = np.copy(w_hat_n_HF)
    w_hat_n_HF = np.copy(w_hat_np1_HF)
    VgradW_hat_nm1_HF = np.copy(VgradW_hat_n_HF)
    
    if np.mod(n, np.round(day/dt)) == 0:
        print('n = ', n, 'of', n_steps)
    
#store the state of the system to allow for a simulation restart at t > 0
if state_store == True:

    keys = ['w_hat_nm1_HF', 'w_hat_n_HF', 'VgradW_hat_nm1_HF']

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
    
#plot vorticity field
fig = plt.figure('w_np1')
ax = fig.add_subplot(111, xlabel=r'x', ylabel=r'y', title='t = ' + str(np.around(t/day, 2)) + ' days')
ct = ax.contourf(x, y, np.fft.irfft2(P*w_hat_np1_HF), 100)
plt.colorbar(ct)
plt.tight_layout()

plt.show()
