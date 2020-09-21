from keras.models import Sequential
from itertools import chain
import matplotlib.pyplot as plt
import easysurrogate as es
import numpy as np
from keras.layers import LSTM, SimpleRNN, Embedding, Dropout
from keras.layers import Dense


def rhs(X_n, s=10, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x = X_n[0]
    y = X_n[1]
    z = X_n[2]

    f_n = np.zeros(3)

    f_n[0] = s * (y - x)
    f_n[1] = r * x - y - x * z
    f_n[2] = x * y - b * z

    return f_n


def step(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)

    # Euler
    X_np1 = X_n + dt * f_n

    return X_np1, f_n


def rhs_surrogate(X_n, s=10):

    feat = feat_eng.get_feat_history(max_lag)
    y_n = model.predict(feat.reshape([1, n_lags, 1]))[0]

    f_n = s * (y_n - X_n)

    return f_n, y_n


def step_surrogate(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n, y_n = rhs_surrogate(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)

    # Euler
    X_np1 = X_n + dt * f_n

    return X_np1, f_n, y_n


plt.close('all')

n_steps = 10**5
dt = 0.01
X = np.zeros(n_steps)
Y = np.zeros(n_steps)
Z = np.zeros(n_steps)
X_dot = np.zeros(n_steps)
Y_dot = np.zeros(n_steps)
Z_dot = np.zeros(n_steps)

# initial condition
X_n = np.zeros(3)
X_n[0] = 0.0
X_n[1] = 1.0
X_n[2] = 1.05

# initial condition right-hand side
f_nm1 = rhs(X_n)

X = np.zeros([n_steps, 3])
X_dot = np.zeros([n_steps, 3])

for n in range(n_steps):

    #step in time
    X_np1, f_n = step(X_n, f_nm1)

    # update variables
    X_n = X_np1
    f_nm1 = f_n

    X[n, :] = X_n
    X_dot[n, :] = f_n

feat_eng = es.methods.Feature_Engineering()
n_increment = 5
lags = [range(0, 50, n_increment)]
n_lags = len(list(chain(*lags)))
max_lag = np.max(list(chain(*lags)))

#X_train[i, -1]
X_train, y_train = feat_eng.lag_training_data([X[:, 0]], X[:, 1], lags)
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], 1])
y_train = y_train.reshape([y_train.shape[0], 1])

n_neurons = 50
# n_feat = X_train.shape[2]
n_train = X_train.shape[0]

model = Sequential()
# model.add(LSTM(n_neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(num_lags, activation='tanh'))
# model.add(Dense(50, activation='tanh'))
# model.add(Dense(8, activation='tanh'))
model.add(SimpleRNN(n_neurons))
# model.add(Dropout(0.01))
# model.add(LSTM(n_neurons))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit network
history = model.fit(X_train, y_train, epochs=10, batch_size=72, verbose=2, shuffle=True)

n_surr = 1000

foo = []

for i in range(n_surr):

    foo.append(model.predict(X_train[i].reshape([1, n_lags, 1]))[0][0])

plt.plot(foo)
plt.plot(y_train[0:n_surr])

# initial condition, pick a random point from the data
idx_start = np.random.randint(0, n_train - max_lag)
idx_start = 0
X_n = X[idx_start, 0]

# feat = []

# features are time lagged, use the data to create initial feature set
for i in range(max_lag + 1):
    feat_eng.append_feat([[X[i, 0]]], max_lag + 1)

X_surr = np.zeros([n_steps, 1])
X_dot_surr = np.zeros([n_steps, 1])

idx = max_lag

for n in range(n_train):

    #step in time
    X_np1, f_n, y_n = step_surrogate(X_n, f_nm1)

    feat_eng.append_feat([[X_np1[0]]], max_lag)

    # update variables
    X_n = X_np1
    f_nm1 = f_n

    X_surr[n, :] = X_n
    X_dot_surr[n, :] = f_n

#############
# Plot PDEs #
#############

post_proc = es.methods.Post_Processing()

print('===============================')
print('Postprocessing results')

fig = plt.figure(figsize=[4, 4])

ax = fig.add_subplot(111, xlabel=r'$x$')
X_dom_surr, X_pde_surr = post_proc.get_pde(X_surr[0:-1:10, 0])
X_dom, X_pde = post_proc.get_pde(X[0:-1:10, 0])
ax.plot(X_dom, X_pde, 'ko', label='L63')
ax.plot(X_dom_surr, X_pde_surr, label='ANN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############
# Plot ACFs #
#############

fig = plt.figure(figsize=[4, 4])

ax = fig.add_subplot(111, ylabel='ACF X', xlabel='time')
R_data = post_proc.auto_correlation_function(X[:, 0], max_lag=500)
R_sol = post_proc.auto_correlation_function(X_surr[:, 0], max_lag=500)
dom_acf = np.arange(R_data.size) * dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

plt.tight_layout()

plt.show()
