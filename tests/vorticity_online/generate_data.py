"""
Generate training data for a reduced subgrid-scale term, applied to 2D Navier Stokes
"""

import matplotlib.pyplot as plt
import easysurrogate as es
import numpy as np
import Vorticity_2D as vort


def draw():
    """
    Draws solution to screen while running the simulation.

    Returns
    -------
    None.

    """
    plt.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.contourf(Q1, 50)
    # ax2.contourf(Q2, 50)
    # ax2.plot(vort_solver_LR.bins, E_spec_LR)
    # ax2.plot(vort_solver_HR.bins, E_spec_HR)
    ax2.plot(campaign.accum_data['Q_LR'])
    ax2.plot(campaign.accum_data['Q_HR'], 'o')
    plt.pause(0.1)
    plt.tight_layout()


def qoi_func(w_hat_n, **kwargs):
    """
    Function which computes the spatially integrated quantities of interest.

    Parameters
    ----------
    w_hat_n : array (complex)
        Vorticity Fourier coefficients.
    **kwargs : array (complex)
        Vorticity Fourier coefficients, passed as a keyowrd argument.

    Returns
    -------
    qoi : array (real)
        The spatiallt-integrated energy and enstrophy.

    """

    n_1d = w_hat_n.shape[0]
    psi_hat_n = kwargs['psi_hat']

    qoi = np.zeros(2)
    qoi[0] = -0.5 * np.dot(psi_hat_n.flatten(), np.conjugate(w_hat_n.flatten())).real / n_1d ** 4
    qoi[1] = 0.5 * np.dot(w_hat_n.flatten(), np.conjugate(w_hat_n.flatten())).real / n_1d ** 4

    return qoi


plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

# spatial resolution low-res model
N_LR = 2**6
# spatial resolution high-res model
N_HR = 2**7
# decay time Fourier mode of viscosity term, at cutoff scale
DECAY_TIME_NU = 5.0
# decay time Fourier mode of  term, at cutoff scale
DECAY_TIME_MU = 90.0
# time step
DT_HR = 0.005
DT_MULTIPLIER = 2
DT_LR = DT_MULTIPLIER * DT_HR

# low-res 2D Navier Stokes solver (in vorticity-stream function formulation)
vort_solver_LR = vort.Vorticity_2D(N_LR, DT_LR, DECAY_TIME_NU, DECAY_TIME_MU)
# high-res 2D Navier Stokes solver
vort_solver_HR = vort.Vorticity_2D(N_HR, DT_HR, DECAY_TIME_NU, DECAY_TIME_MU)

# create an EasySurrogate campaign
campaign = es.Campaign()
# the number of QoI to track
N_Q = 2
# create a reduced SGS surrogate object
surrogate = es.methods.Reduced_Surrogate(N_Q, N_LR)
# add the surrogate to the campaign
campaign.add_app('reduced_vort', surrogate)

# start, end time (in days)
DAY = vort_solver_LR.day
T = 500 * DAY
T_END = T + 10 * 365 * DAY
n_steps = np.ceil((T_END - T) / DT_LR).astype('int')

# plot the solution while running
PLOT = False
plot_frame_rate = np.floor(DAY / DT_LR).astype('int')
# restart from a previous stored state
RESTART = True
# store accumulated data at the end
STORE_DATA = True

if PLOT:
    fig = plt.figure(figsize=[8, 4])

if RESTART:
    # load the HR state from a HDF5 file
    IC_HR = campaign.load_hdf5_data(
        file_path='./restart/state_HR_t%d_N%d.hdf5' %
        (T / DAY, DT_MULTIPLIER))
    # HR vorticity at HR stencil k
    w_hat_k_HR = IC_HR['w_hat_k_HR']
    # HR vorticity at HR stencil k-1
    w_hat_km1_HR = IC_HR['w_hat_km1_HR']
    # HR Jacobian at HR stencil n-1
    VgradW_hat_km1_HR = IC_HR['VgradW_hat_km1_HR']
    # HR vorticity at LR stencil n
    w_hat_n_HR = IC_HR['w_hat_n_HR']
    # HR vorticity at LR stencil n - 1
    w_hat_nm1_HR = IC_HR['w_hat_nm1_HR']

    # load the LR state from a HDF5 file
    IC_LR = campaign.load_hdf5_data(
        file_path='./restart/state_LR_t%d_N%d.hdf5' %
        (T / DAY, DT_MULTIPLIER))
    w_hat_n_LR = IC_LR['w_hat_n_LR']
    w_hat_nm1_LR = IC_LR['w_hat_nm1_LR']
    VgradW_hat_nm1_LR = IC_LR['VgradW_hat_nm1_LR']
else:
    # compute the HR initial condition stencil vorticity at HR step k and k-1
    # and Jacobian at k-1
    w_hat_k_HR, w_hat_km1_HR, VgradW_hat_km1_HR = vort_solver_HR.inital_cond()
    # the corresponding HR values at the times of the LR stencil
    w_hat_n_HR = w_hat_k_HR
    w_hat_nm1_HR = w_hat_km1_HR
    # compute the LR initial condition stencil vorticity at LR step n and n-1
    # and Jacobian at n-1
    w_hat_n_LR, w_hat_nm1_LR, VgradW_hat_nm1_LR = vort_solver_LR.inital_cond()

# time loop
for n in range(n_steps):

    if np.mod(n, int(10 * DAY / DT_LR)) == 0:
        print('%.1f percent complete' % (n / n_steps * 100))

    # integrate the HR solver over DT_MULTIPLIER HR time steps
    for i in range(DT_MULTIPLIER):
        w_hat_kp1_HR, VgradW_hat_k_HR = vort_solver_HR.step(w_hat_k_HR, w_hat_km1_HR,
                                                            VgradW_hat_km1_HR)
        # update HR vars
        w_hat_km1_HR = np.copy(w_hat_k_HR)
        w_hat_k_HR = np.copy(w_hat_kp1_HR)
        VgradW_hat_km1_HR = np.copy(VgradW_hat_k_HR)

    # the HR vorticity at time t_n = n * DT_LR
    w_hat_np1_HR = w_hat_kp1_HR

    # exact sgs term
    # sgs_hat_exact = vort_solver_HR.down_scale(VgradW_hat_nm1_HR, N_LR) - VgradW_hat_nm1_LR

    # compute the HR stream function
    psi_hat_n_HR = vort_solver_HR.compute_stream_function(w_hat_n_HR)

    # compute the LR stream function
    psi_hat_n_LR = vort_solver_LR.compute_stream_function(w_hat_n_LR)

    # compute the QoI using the HR state
    Q_HR = qoi_func(w_hat_n_HR, psi_hat=psi_hat_n_HR)

    # compute the QoI using the LR state
    Q_LR = qoi_func(w_hat_n_LR, psi_hat=psi_hat_n_LR)

    # compute the reduced subgrid-scale term
    reduced_dict = surrogate.train([-psi_hat_n_LR, w_hat_n_LR], Q_HR - Q_LR)
    sgs_hat_reduced = reduced_dict['sgs_hat']

    # integrate the LR solver with reduced sgs term
    w_hat_np1_LR, VgradW_hat_n_LR = vort_solver_LR.step(w_hat_n_LR, w_hat_nm1_LR,
                                                        VgradW_hat_nm1_LR,
                                                        sgs_hat=sgs_hat_reduced)

    # accumulate QoI data inside the EasySurrogate campaign
    campaign.accumulate_data({'Q_LR': Q_LR, 'Q_HR': Q_HR})
    del reduced_dict['sgs_hat']
    campaign.accumulate_data(reduced_dict)

    # update LR vars
    w_hat_nm1_LR = w_hat_n_LR
    w_hat_n_LR = w_hat_np1_LR
    VgradW_hat_nm1_LR = VgradW_hat_n_LR

    # update HR vars on LR time stencil
    w_hat_nm1_HR = w_hat_n_HR
    w_hat_n_HR = w_hat_np1_HR

    T += DT_LR

    # PLOT solution while running
    if np.mod(n, plot_frame_rate) == 0 and PLOT:
        Q1 = np.fft.ifft2(w_hat_np1_HR).real
        # Q2 = np.fft.ifft2(w_hat_np1_LR).real
        # sgs = np.fft.ifft2(sgs_hat_reduced).real
        # E_spec_LR, Z_spec_LR = vort_solver_LR.spectrum(w_hat_np1_LR)
        # E_spec_HR, Z_spec_HR = vort_solver_HR.spectrum(w_hat_np1_HR)
        draw()

# store the state of the LR and HR models
campaign.store_data_to_hdf5({'w_hat_n_HR': w_hat_n_HR, 'w_hat_nm1_HR': w_hat_nm1_HR,
                             'w_hat_k_HR': w_hat_k_HR, 'w_hat_km1_HR': w_hat_km1_HR,
                             'VgradW_hat_km1_HR': VgradW_hat_km1_HR},
                            file_path='./restart/state_HR_t%d_N%d.hdf5' % (T / DAY, DT_MULTIPLIER))
campaign.store_data_to_hdf5({'w_hat_np1_LR': w_hat_np1_LR, 'w_hat_n_LR': w_hat_n_LR,
                             'w_hat_nm1_LR': w_hat_nm1_LR,
                             'VgradW_hat_nm1_LR': VgradW_hat_nm1_LR},
                            file_path='./restart/state_LR_t%d_N%d.hdf5' % (T / DAY, DT_MULTIPLIER))

# store the accumulated data
if STORE_DATA:
    campaign.store_accumulated_data(file_path='../samples/reduced_vorticity_training_N%d.hdf5' %
                                    (DT_MULTIPLIER))
