import easysurrogate as es
import L96 as solver
import numpy as np

#time step
dt = 0.01

#create a solver for the two-layer Lorenz 96 model
l96 = solver.L96(dt, h_x=-2)

#simulation time
t_end = 1000.0
t = np.arange(0.0, t_end, dt)

#load a pre-trained surrogate
campaign = es.Campaign(load_state=True)

# #get the initial condition of X and the right-hand side of the X ODE
X_n, f_nm1 = l96.initial_conditions()
r_n = np.zeros(l96.K)

# start time integration
for idx, t_i in enumerate(t):
    
    if idx > campaign.surrogate.max_lag:
        # Predict the subgrid scale term with the QSN surrogate
        r_n = campaign.surrogate.predict(X_n)

    # Integrate the two-layer model in time
    X_np1, f_n = l96.step(X_n, f_nm1, r_n=r_n)

    # Update variables
    X_n = X_np1
    f_nm1 = f_n
    
    # Store data
    snapshot = {'X_n': X_n, 'r_n': r_n}
    campaign.accumulate_data(snapshot)

    if np.mod(idx, 1000) == 0:
        print('t =', np.around(t_i, 1), 'of', t_end)

#store the accumulated data to an HDF5 file
campaign.store_accumulated_data()

#plot the X solution and the subgrid-scale term at the final time
l96.plot_solution()
