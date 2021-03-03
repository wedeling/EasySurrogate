import numpy as np
import Vorticity_2D as vort
import matplotlib.pyplot as plt
import easysurrogate as es


def draw():
    ax1.contourf(Q1, 50)
    # ax2.contourf(Q2, 50)
    ax2.plot(vort_solver_LR.bins, E_spec_LR)
    ax2.plot(vort_solver_HR.bins, E_spec_HR)
    plt.pause(0.1)
    plt.tight_layout()
    
def compute_int(X1_hat, X2_hat, N):
    """
    Compute the integral of X1*X2 using the Fourier expansion
    """
    integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten())) / N**4
    return integral.real


plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

N_LR = 2**6
N_HR = 2**7
decay_time_nu = 5.0
decay_time_mu = 90.0
dt = 0.01

vort_solver_LR = vort.Vorticity_2D(N_LR, dt, decay_time_nu, decay_time_mu)
vort_solver_HR = vort.Vorticity_2D(N_HR, dt, decay_time_nu, decay_time_mu)

campaign = es.Campaign()
#
N_Q = 2
# create a reduced SGS surrogate object
surrogate = es.methods.Reduced_Surrogate(N_Q, N_LR)
campaign.add_app('reduced_vort', surrogate)

# start, end time (in days) + time step
day = vort_solver_LR.day
t_burn = 365 * day
t = 0 * day
t_end = t + 50 * day
n_steps = np.ceil((t_end - t) / dt).astype('int')

plot = True
store_frame_rate = np.floor(day / dt).astype('int')

if plot:
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, yscale='log')

w_hat_n_HR, w_hat_nm1_HR, VgradW_hat_nm1_HR = vort_solver_HR.inital_cond()
w_hat_n_LR, w_hat_nm1_LR, VgradW_hat_nm1_LR = vort_solver_LR.inital_cond()

j = 0

# time loop
for n in range(n_steps):
    w_hat_np1_HR, VgradW_hat_n_HR = vort_solver_HR.step(w_hat_n_HR, w_hat_nm1_HR,
                                                        VgradW_hat_nm1_HR)

    # exact sgs term
    sgs_hat = vort_solver_HR.down_scale(VgradW_hat_nm1_HR, N_LR) - VgradW_hat_nm1_LR
    
    w_hat_n_HR_projected = vort_solver_HR.down_scale(w_hat_n_HR, N_LR)
    psi_hat_n_HR_projected = vort_solver_LR.compute_stream_function(w_hat_n_HR_projected)
    Q_HR = np.zeros(N_Q)
    Q_HR[0] = -0.5 * compute_int(psi_hat_n_HR_projected, w_hat_n_HR_projected, N_LR)
    Q_HR[1] = 0.5 * compute_int(w_hat_n_HR_projected, w_hat_n_HR_projected, N_LR)
    
    psi_hat_n_LR = vort_solver_LR.compute_stream_function(w_hat_n_LR)
    Q_LR = np.zeros(N_Q)
    Q_LR[0] = -0.5 * compute_int(psi_hat_n_LR, w_hat_n_LR, N_LR)
    Q_LR[1] = 0.5 * compute_int(w_hat_n_LR, w_hat_n_LR, N_LR)

    print(Q_LR)
    print(Q_HR)
    print('--------')

    reduced_dict_u = surrogate.train([psi_hat_n_LR, w_hat_n_LR], Q_HR - Q_LR)

    w_hat_np1_LR, VgradW_hat_n_LR = vort_solver_LR.step(w_hat_n_LR, w_hat_nm1_LR,
                                                        VgradW_hat_nm1_LR,
                                                        sgs_hat=sgs_hat)

    # update vars
    w_hat_nm1_HR = np.copy(w_hat_n_HR)
    w_hat_n_HR = np.copy(w_hat_np1_HR)
    VgradW_hat_nm1_HR = np.copy(VgradW_hat_n_HR)

    w_hat_nm1_LR = np.copy(w_hat_n_LR)
    w_hat_n_LR = np.copy(w_hat_np1_LR)
    VgradW_hat_nm1_LR = np.copy(VgradW_hat_n_LR)

    j += 1

    if np.mod(j, store_frame_rate) == 0:
        Q1 = np.fft.ifft2(w_hat_np1_HR).real
        Q2 = np.fft.ifft2(w_hat_np1_LR).real
        E_spec_LR, Z_spec_LR = vort_solver_LR.spectrum(w_hat_np1_LR) 
        E_spec_HR, Z_spec_HR = vort_solver_HR.spectrum(w_hat_np1_HR) 
        draw()
        j = 0
