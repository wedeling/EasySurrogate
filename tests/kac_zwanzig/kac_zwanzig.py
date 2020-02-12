"""
===============================================================================
Integrate full Kac-Zwanzig model to generate data
-------------------------------------------------------------------------------
 Follows set-up from Stuart and Warren, J. Stat. Phys. 1999
===============================================================================
"""

def step(v_n, p_n, u_n, q_n):
    """
    
    Integrates v, p, u, q in time via sympletic Euler
    
    Parameters
    ----------
    u : (array of floats): position heat bath particles.
    p : (float): position distinguished particle.
    v : (array of floats): momentum heat bath particles 
    q : (float): momentum distinguished particle

    Returns
    -------
    
    v, p, u, q at next time level
    
    """
    
    v_np1 = v_n - dt*j2*(u_n - q_n)
    # potential V(q)=1/4 * (q^2-1)^2
    r_n = np.mean(u_n)
    p_np1 = p_n - dt*q_n*(q_n**2 - 1) + dt*G**2*N*(r_n - q_n)
    u_np1 = u_n + dt*v_np1
    q_np1 = q_n + dt*p_np1
    
    #Hamiltonian
    H_n = 0.5*p_n**2 + 0.25*(q_n**2 - 1)**2 + np.sum(0.5*v_n**2*j2 + 0.5*(u_n - q_n)**2)
    
    return v_np1, p_np1, u_np1, q_np1, r_n, H_n

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

# Number of heat bath particles
N = 100

# Output every Nskip integration steps
N_skip = 1

# Number of output data points
M = 10**6

# Coupling and mass scaling
G = 1.0

# Inverse temperature
beta = 0.0001   

# Frequencies (sqaured) heath bath particles
j2 = np.arange(1, N+1)**2

##################
# Initialisation #
##################

# distinguished particle:
q_n = 1  # position
p_n = 0  # momentum

# heat bath particles:
u_i = np.random.randn(N)

# position
u_n = u_i/(G*np.sqrt(beta))

# momentum
v_n = np.zeros(N)

#####################################
# Integration with symplectic Euler #
#####################################

# % delT = output time step, delT = dt*Nskip = Nskip*0.01/N

dt = 0.01/N  # integration time step
delT = dt*N_skip  # output time step

# data to store
QoI = ['q_n', 'p_n', 'r_n', 'H_n']
Size = [1, 1, 1, 1]
data = {}
idx = 0
for qoi in QoI:
    data[qoi] = np.zeros([M, Size[idx]])
    idx += 1

# initial Nskip*10^3 integration steps are discarded as transient
for j in range(int(N_skip*1e3)):
    v_np1, p_np1, u_np1, q_np1, r_n, H_n = step(v_n, p_n, u_n, q_n)

    v_n = v_np1
    p_n = p_np1
    u_n = u_np1
    q_n = q_np1

idx = 0
j = 0
for i in range(M*N_skip):
    v_np1, p_np1, u_np1, q_np1, r_n, H_n = step(v_n, p_n, u_n, q_n)
    idx += 1

    if np.mod(idx, N_skip) == 0:
        idx = 0
        for qoi in QoI:
            data[qoi][j,:] = eval(qoi)
        j += 1

    v_n = v_np1
    p_n = p_np1
    u_n = u_np1
    q_n = q_np1

# Store data
post_proc = es.methods.Post_Processing()
post_proc.store_samples_hdf5(data)
    
#############
# Plot PDEs #
#############

dom_q, pdf_q = post_proc.get_pde(data['q_n'][0:-1:10])
dom_p, pdf_p = post_proc.get_pde(data['p_n'][0:-1:10])
dom_r, pdf_r = post_proc.get_pde(data['r_n'][0:-1:10])

fig = plt.figure(figsize=[12,4])
ax1 = fig.add_subplot(131, xlabel = r'position q', yticks = [])
ax2 = fig.add_subplot(132, xlabel = r'momentum p', yticks = [])
ax3 = fig.add_subplot(133, xlabel = r'sgs r', yticks = [])

ax1.plot(dom_q, pdf_q)
ax2.plot(dom_p, pdf_p)
ax3.plot(dom_r, pdf_r)

plt.tight_layout()

############
# plot ACF #
############

# plt.figure()
# acf = post_proc.auto_correlation_function(data['r_n'], 50000)
# plt.plot(acf)

plt.show()