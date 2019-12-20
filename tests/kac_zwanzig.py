"""
===============================================================================
Integrate full Kac-Zwanzig model to generate data
-------------------------------------------------------------------------------
 Follows set-up from Stuart and Warren, J. Stat. Phys. 1999
===============================================================================
"""

def step(v, p, u, q):
    """
    
    Integrates v, p, u, q in time via sympletic Euler
    
    Parameters
    ----------
    v : (array of floats): position heat bath particles.
    p : (float): position distinguished particle.
    u : (array of floats): momentum heat bath particles 
    q : (float): momentum distinguished particle

    Returns
    -------
    
    v, p, u, q at next time level
    
    """
    
    v = v - dt*j2*(u - q)
    # potential V(q)=1/4 * (q^2-1)^2
    p = p - dt*q*(q**2 - 1) + dt*G**2*(np.sum(u) - N*q)
    u = u + dt*v
    q = q + dt*p
    
    return v, p, u, q

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

# Number of heat bath particles
N = 100

# Output every Nskip integration steps
N_skip = 100

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
q = 1  # position
p = 0  # momentum

# heat bath particles:
u_i = np.random.randn(N)

# momentum
u = u_i/(G*np.sqrt(beta))

# position
v = np.zeros(N)

#####################################
# Integration with symplectic Euler #
#####################################

# % delT = output time step, delT = dt*Nskip = Nskip*0.01/N

dt = 0.01/N  # integration time step
delT = dt*N_skip  # output time step

# data to store
QoI = ['q', 'p', 'r']
data = {}
for qoi in QoI:
    data[qoi] = np.zeros(M)

# initial Nskip*10^3 integration steps are discarded as transient
for j in range(int(N_skip*1e3)):
   v, p, u, q = step(v, p, u, q)

for i in range(M):
    for j in range(N_skip):
        v, p, u, q = step(v, p, u, q)
        r = np.sum(u)

    for qoi in QoI:
        data[qoi][i] = eval(qoi)

# Store data
post_proc = es.methods.Post_Processing()
post_proc.store_samples_hdf5(data)
    
#############
# Plot PDEs #
#############

dom_q, pdf_q = post_proc.get_pde(data['q'])
dom_p, pdf_p = post_proc.get_pde(data['p'])

fig = plt.figure(figsize=[8,4])
ax1 = fig.add_subplot(121, xlabel = r'position q', yticks = [])
ax2 = fig.add_subplot(122, xlabel = r'momentum p', yticks = [])

ax1.plot(dom_q, pdf_q)
ax2.plot(dom_p, pdf_p)

plt.tight_layout()

plt.show()