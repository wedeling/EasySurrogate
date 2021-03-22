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
    # ax2.plot(campaign.accum_data['Q_LR'])
    # ax2.plot(campaign.accum_data['Q_HR'], 'o')
    ax2.plot(np.array(plot1))
    ax2.plot(np.array(plot2))
    plt.pause(0.1)
    plt.tight_layout()


def compute_int(x1_hat, x2_hat, n_points_1d):
    """
    Compute the integral of X1*X2 using the Fourier expansion
    """
    integral = np.dot(x1_hat.flatten(), np.conjugate(x2_hat.flatten())) / n_points_1d**4
    return integral.real

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
        The spatially-integrated energy and enstrophy.

    """
    
    n_1d = w_hat_n.shape[0]
    if n_1d == N_LR:
        psi_hat_n = kwargs['psi_hat_n_LR']
    else:
        psi_hat_n = kwargs['psi_hat_n_HR']

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
DT = 0.01

# low-res 2D Navier Stokes solver (in vorticity-stream function formulation)
vort_solver_LR = vort.Vorticity_2D(N_LR, DT, DECAY_TIME_NU, DECAY_TIME_MU)
# high-res 2D Navier Stokes solver
vort_solver_HR = vort.Vorticity_2D(N_HR, DT, DECAY_TIME_NU, DECAY_TIME_MU)

# create an EasySurrogate campaign
campaign = es.Campaign()
# the number of QoI to track
N_Q = 2
# create a reduced SGS surrogate object
surrogate = es.methods.Reduced_Surrogate(N_Q, N_LR)
# add the surrogate to the campaign
campaign.add_app('reduced_vort', surrogate)

# load the dQ surrogate
dQ_campaign = es.Campaign(load_state=True)
ann = dQ_campaign.surrogate

# store the surrogate used to predict dQ
campaign.surrogate.set_dQ_surr(ann)
# online learning nudging time scale
TAU_NUDGE = 1.0
# length of the moving window (in time steps) in which the online training data is stored
WINDOW_LENGTH = 2
# batch size used in inline learning
BATCH_SIZE = 1
# number of time steps to wait before predicting dQ with the surrogate
SETTLING_PERIOD = 1
#
M = 10
# set the online learning parameters
campaign.surrogate.set_online_training_parameters(TAU_NUDGE, DT, WINDOW_LENGTH)

#the indices of a upper triangular N_Q x N_Q array
idx1, idx2 = np.triu_indices(N_Q)

# the reference data frame used to train the ANN
data_frame_ref = campaign.load_hdf5_data(file_path='../samples/reduced_vorticity_training2.hdf5')
dQ_ref = data_frame_ref['Q_HR'] - data_frame_ref['Q_LR']
inner_prods = data_frame_ref['inner_prods'][0][idx1, idx2]
c_ij = data_frame_ref['c_ij'][0].flatten()
src_Q = data_frame_ref['src_Q'][0]

# start, end time (in days)
DAY = vort_solver_LR.day
T = 500 * DAY
T_END = T + 10 * 365 * DAY
n_steps = np.ceil((T_END - T) / DT).astype('int')

# PLOT the solution while running
PLOT = True
plot_frame_rate = np.floor(DAY / DT).astype('int')
# RESTART from a previous stored state
RESTART = True
# store the data at the end
STORE = True

if PLOT:
    fig = plt.figure(figsize=[8, 4])
    plot1 = []
    plot2 = []

if RESTART:
    # load the HR state from a HDF5 file
    IC_HR = campaign.load_hdf5_data(file_path='./restart/state_HR_t%d.hdf5' % (T / DAY))
    w_hat_n_HR = IC_HR['w_hat_n_HR']
    w_hat_nm1_HR = IC_HR['w_hat_nm1_HR']
    VgradW_hat_nm1_HR = IC_HR['VgradW_hat_nm1_HR']
    # load the LR state from a HDF5 file
    IC_LR = campaign.load_hdf5_data(file_path='./restart/state_LR_t%d.hdf5' % (T / DAY))
    w_hat_n_LR = IC_LR['w_hat_n_LR']
    w_hat_nm1_LR = IC_LR['w_hat_nm1_LR']
    VgradW_hat_nm1_LR = IC_LR['VgradW_hat_nm1_LR']
else:
    # compute the initial condition
    w_hat_n_HR, w_hat_nm1_HR, VgradW_hat_nm1_HR = vort_solver_HR.inital_cond()
    w_hat_n_LR, w_hat_nm1_LR, VgradW_hat_nm1_LR = vort_solver_LR.inital_cond()

# time loop
for n in range(n_steps):

    if np.mod(n, int(10 * DAY / DT)) == 0:
        print('%.1f percent complete' % (n / n_steps * 100))

    # compute the difference between the LR and the HR states. To do so,
    # the LR state muts be upscaled to the HR grid.
    Delta_w_hat = campaign.surrogate.up_scale(VgradW_hat_nm1_LR, N_HR) - VgradW_hat_nm1_HR
    
    # integrate the HR solver
    w_hat_np1_HR, VgradW_hat_n_HR = vort_solver_HR.step(w_hat_n_HR, w_hat_nm1_HR,
                                                        VgradW_hat_nm1_HR)#,
                                                        # sgs_hat = Delta_w_hat / TAU_NUDGE)

    # exact sgs term
    # sgs_hat_exact = vort_solver_HR.down_scale(VgradW_hat_nm1_HR, N_LR) - VgradW_hat_nm1_LR

    # compute the HR stream function
    psi_hat_n_HR = vort_solver_HR.compute_stream_function(w_hat_n_HR)

    # compute the LR stream function
    psi_hat_n_LR = vort_solver_LR.compute_stream_function(w_hat_n_LR)

    # compute the QoI using the HR state
    Q_HR = qoi_func(w_hat_n_HR, psi_hat_n_HR = psi_hat_n_HR)

    # compute the QoI using the LR state
    Q_LR = qoi_func(w_hat_n_LR, psi_hat_n_LR = psi_hat_n_LR)

    plot1.append(Q_HR)
    plot2.append(Q_LR)
    
    # make a prediction for the reduced sgs terms
    if n > SETTLING_PERIOD:
        # predict dQ using the surrogate model
        dQ_pred = ann.predict([inner_prods, c_ij, src_Q])
    else:
        # just the reference data in the settling period
        dQ_pred = dQ_ref[n]

    # compute the reduced subgrid-scale term
    reduced_dict = surrogate.train([-psi_hat_n_LR, w_hat_n_LR], dQ_pred)
    sgs_hat_reduced = reduced_dict['sgs_hat']

    # collect features
    inner_prods = reduced_dict['inner_prods'][idx1, idx2]
    c_ij = reduced_dict['c_ij'].flatten()
    src_Q = reduced_dict['src_Q']

    # features
    feats = [inner_prods, c_ij, src_Q]

    # Generate the training data for the online learning back prop step
    campaign.surrogate.generate_online_training_data(feats, w_hat_nm1_LR, w_hat_n_LR,
                                                     w_hat_nm1_HR, w_hat_n_HR,
                                                     qoi_func,
                                                     #these are the kwargs needed in qoi_func
                                                     psi_hat_n_LR = psi_hat_n_LR,
                                                     psi_hat_n_HR = psi_hat_n_HR)

    # update the neural network for dQ every M time steps
    if np.mod(n, M) == 0 and n > SETTLING_PERIOD and n > WINDOW_LENGTH:
        ann.train_online(batch_size=BATCH_SIZE, verbose=True)

    # integrate the LR solver with reduced sgs term
    w_hat_np1_LR, VgradW_hat_n_LR = vort_solver_LR.step(w_hat_n_LR, w_hat_nm1_LR,
                                                        VgradW_hat_nm1_LR,
                                                        sgs_hat=sgs_hat_reduced)

    # accumulate QoI data inside the EasySurrogate campaign
    campaign.accumulate_data({'Q_LR': Q_LR, 'Q_HR': Q_HR})
    del reduced_dict['sgs_hat']
    campaign.accumulate_data(reduced_dict)

    # update vars
    w_hat_nm1_HR = np.copy(w_hat_n_HR)
    w_hat_n_HR = np.copy(w_hat_np1_HR)
    VgradW_hat_nm1_HR = np.copy(VgradW_hat_n_HR)

    w_hat_nm1_LR = np.copy(w_hat_n_LR)
    w_hat_n_LR = np.copy(w_hat_np1_LR)
    VgradW_hat_nm1_LR = np.copy(VgradW_hat_n_LR)

    T += DT

    # plot solution while running
    if np.mod(n, plot_frame_rate) == 0 and PLOT:
        Q1 = np.fft.ifft2(w_hat_np1_HR).real
        # Delta_w = np.fft.ifft2(Delta_w_hat).real
        Q2 = np.fft.ifft2(w_hat_np1_LR).real
        # sgs = np.fft.ifft2(sgs_hat_reduced).real
        # E_spec_LR, Z_spec_LR = vort_solver_LR.spectrum(w_hat_np1_LR)
        # E_spec_HR, Z_spec_HR = vort_solver_HR.spectrum(w_hat_np1_HR)
        draw()

# store the state of the LR and HR models
campaign.store_data_to_hdf5({'w_hat_n_HR': w_hat_n_HR, 'w_hat_nm1_HR': w_hat_nm1_HR,
                             'VgradW_hat_nm1_HR': VgradW_hat_nm1_HR},
                            file_path='./restart/state_HR_t%d.hdf5' % (T / DAY))
campaign.store_data_to_hdf5({'w_hat_n_LR': w_hat_n_LR, 'w_hat_nm1_LR': w_hat_nm1_LR,
                             'VgradW_hat_nm1_LR': VgradW_hat_nm1_LR},
                            file_path='./restart/state_LR_t%d.hdf5' % (T / DAY))

# store the accumulated data
if STORE:
    campaign.store_accumulated_data(file_path='../samples/reduced_vorticity_online2.hdf5')
