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

def get_P_full(cutoff):

    P = np.ones([N, N])

    for i in range(N):
        for j in range(N):

            if np.abs(kx_full[i, j]) > cutoff or np.abs(ky_full[i, j]) > cutoff:
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

def freq_map():
    """
    Map 2D frequencies to a 1D bin (kx, ky) --> k
    where k = 0, 1, ..., sqrt(2)*Ncutoff
    """
   
    #edges of 1D wavenumber bins
    bins = np.arange(-0.5, np.ceil(2**0.5*Ncutoff)+1)
    #fmap = np.zeros([N,N]).astype('int')
    
    dist = np.zeros([N,N])
    
    for i in range(N):
        for j in range(N):
            #Euclidian distance of frequencies kx and ky
            dist[i, j] = np.sqrt(kx_full[i,j]**2 + ky_full[i,j]**2).imag
                
    #find 1D bin index of dist
    _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(N**2), bins=bins)
    
    binnumbers -= 1
            
    return binnumbers.reshape([N, N]), bins

def spectrum(w_hat, P):

    #convert rfft2 coefficients to fft2 coefficients
    w_hat_full = np.zeros([N, N]) + 0.0j
    w_hat_full[0:N, 0:int(N/2+1)] = w_hat
    w_hat_full[map_I, map_J] = np.conjugate(w_hat[I, J])
    w_hat_full *= P
    
    psi_hat_full = w_hat_full/k_squared_no_zero_full
    psi_hat_full[0,0] = 0.0
    
    E_hat = -0.5*psi_hat_full*np.conjugate(w_hat_full)/N**4
    Z_hat = 0.5*w_hat_full*np.conjugate(w_hat_full)/N**4
    
    E_spec = np.zeros(N_bins)
    Z_spec = np.zeros(N_bins)
    
    for i in range(N):
        for j in range(N):
            bin_idx = binnumbers[i, j]
            E_spec[bin_idx] += E_hat[i, j].real
            Z_spec[bin_idx] += Z_hat[i, j].real
            
    return E_spec, Z_spec
"""
***************************
* M A I N   P R O G R A M *
***************************
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = 6,6
import os, h5py
import easysurrogate as es
from scipy import stats 

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

kx_full = np.zeros([N, N]) + 0.0j
ky_full = np.zeros([N, N]) + 0.0j

for i in range(N):
    for j in range(N):
        kx_full[i, j] = 1j*k[j]
        ky_full[i, j] = 1j*k[i]

k_squared_full = kx_full**2 + ky_full**2
k_squared_no_zero_full = np.copy(k_squared_full)
k_squared_no_zero_full[0,0] = 1.0

#cutoff in pseudospectral method
Ncutoff = N/3
Ncutoff_LF = N_LF/3

shift = np.zeros(N).astype('int')
for i in range(1,N):
    shift[i] = np.int(N-i)
I = range(N);J = range(np.int(N/2+1))
map_I, map_J = np.meshgrid(shift[I], shift[J])
I, J = np.meshgrid(I, J)

binnumbers, bins = freq_map()
N_bins = bins.size

#spectral filter
P = get_P(Ncutoff)
P_LF = get_P(Ncutoff_LF)
#spectral filter for the full FFT2 
P_full = get_P_full(Ncutoff)
P_LF_full = get_P_full(Ncutoff_LF)

#filter out all the zeros from the 2/3 rule
K, L = np.where(P_LF == 1.0)

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
t_end = t + 2*365*day
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

vorticity = np.zeros([N**2,S])
jacobian = np.zeros([N**2,S])

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
            samples[QoI[q]] = np.zeros([S, 2*np.int(Ncutoff_LF)+1, np.int(Ncutoff_LF)+1],dtype='complex') 
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
                #store non-zero Fourier coefs
                samples[qoi][idx] = eval(qoi)[K,L].reshape(2*np.int(Ncutoff_LF)+1,np.int(Ncutoff_LF)+1)
                #store full fields
                # samples[qoi][idx] = np.fft.irfft2(eval(qoi))

            vorticity[:,idx] = np.fft.irfft2(w_hat_n_HF).flatten()
            jacobian[:,idx] = np.fft.irfft2(VgradW_hat_nm1_HF).flatten()

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
    
post_proc = es.methods.Post_Processing()
# plot PDFs
w_dom, w_pdf = post_proc.get_pdf(vorticity.flatten())
J_dom, J_pdf = post_proc.get_pdf(jacobian.flatten())

fig = plt.figure(figsize=[12,6])
ax1 = fig.add_subplot(121, xlabel=r'w')
ax1.plot(w_dom,w_pdf,lw=2,label='vorticity')
ax2 = fig.add_subplot(122, xlabel=r'J')
ax2.plot(J_dom,J_pdf,lw=2,label='Jacobian')
plt.tight_layout()

# compute energy and enstrophy spectrum
ehat_HF, zhat_HF = spectrum(w_hat_np1_HF,P_full)
ehat_LF, zhat_LF = spectrum(w_hat_np1_LF,P_LF_full)

fig = plt.figure(figsize=[12,6])
ax1 = fig.add_subplot(121, xlabel=r'wavenumber',ylabel=r'Energy spectra')
ax1.loglog(ehat_HF,lw=2,label='HF')
ax1.loglog(ehat_LF,lw=2,label='LF')
ax1.legend(loc='best')
ax2 = fig.add_subplot(122, xlabel=r'wavenumber',ylabel=r'Enstrophy spectra')
ax2.loglog(zhat_HF,lw=2,label='HF')
ax2.loglog(zhat_LF,lw=2,label='LF')
ax2.legend(loc='best')
plt.tight_layout()

if plot == False:
    #plot vorticity field
    fig = plt.figure(figsize=[12,6])
    ax = fig.add_subplot(121, xlabel=r'x', ylabel=r'y', title='t = ' + str(np.around(t/day, 2)) + ' days')
    ct = ax.contourf(x, y, np.fft.irfft2(w_hat_np1_HF), 100)
    plt.colorbar(ct)
    
    ax = fig.add_subplot(122, xlabel=r'x', ylabel=r'y', title='t = ' + str(np.around(t/day, 2)) + ' days')
    ct = ax.contourf(x, y, np.fft.irfft2(P_LF*w_hat_np1_LF), 100)
    plt.colorbar(ct)
    
    plt.tight_layout()
    
    plt.show()