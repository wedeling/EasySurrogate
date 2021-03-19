import numpy as np
import matplotlib.pyplot as plt


def down_scale(X_hat, N_LR):
    """
    Down-scale X to a lower spatial resolution by removing high-frequency Fourier coefficients.

    Parameters
    ----------
    X_hat : array (complex)
        An array of Fourier coefficients of X, where the spatial resolution in 1D is higher than N.
    N_LR : int
        The new, lower, spatial resolution (X.shape[0] < N).

    Returns
    -------
    X_hat : array (complex)
        The Fourier coefficients of X at a spatial resolution determined by N.

    """
    assert N_LR < X_hat.shape[0], "N must be smaller than X_hat.shape[0] to down scale."

    # The spatial dimension of the problem
    d = X_hat.ndim
    # The HR grid resolution
    N_HR = X_hat.shape[0]

    # the range that should be deleted
    start = int(N_LR / 2)
    end = X_hat.shape[0] - start
    # Remove the Fourier coefficients that are not present in the lower-resolution version
    # of X_hat
    for i in range(X_hat.ndim):
        X_hat = np.delete(X_hat, np.arange(start, end), axis=i)
    # In numpy only the inverse transform is scaled. The following term must be applied
    # to ensure the correct scaling factor is applied in the inverse transform.
    scaling_factor = (N_LR / N_HR)**d
    return X_hat * scaling_factor


def up_scale(X_hat, N_HR):
    """
    Up-scale X to a higher spatial resolution by padding high-frequency Fourier coefficients with
    zeros. Thus far this will only work for 1 or 2 dimensional arrays.


    Parameters
    ----------
    X_hat : array (complex)
        The Fourier coefficients of X.
    N_HR : int
        The new, higher, spatial resolution (X.shape[0] > N).

    Returns
    -------
    X_hat : array (complex)
        The Fourier coefficients of X at a higher spatial resolution determined ny N.

    """

    N_LR = X_hat.shape[0]
    d = X_hat.ndim

    assert N_LR < N_HR, "X_hat.shape[0] must be < N_HR in order to upscale X_hat."
    assert d == 1 or d == 2, "Upscaling only implemented for 1d or 2d arrays."

    start = int(N_LR / 2)
    pad_size = N_HR - N_LR
    if X_hat.ndim == 1:
        # pad the 1d array with zeros
        X_hat = np.insert(X_hat, start, np.zeros(pad_size) + 0j)
    elif X_hat.ndim == 2:
        # pad the 2d array with a 'cross' of zeros
        X_hat = np.insert(X_hat, start, np.zeros([pad_size, N_LR]) + 0j, axis=0)
        X_hat = np.insert(X_hat, start, np.zeros([pad_size, N_HR]) + 0j, axis=1)
    # In numpy only the inverse transform is scaled. The following term must be applied
    # to ensure the correct scaling factor is applied in the inverse transform.
    scaling_factor = (N_HR / N_LR)**d
    return X_hat * scaling_factor


def initial_cond(N):
    """
    Compute the initial condition

    Returns
    -------
    u_hat, v_hat: array(complex)
        initial Fourier coefficients of u and v

    """

    xx, yy = get_grid(1.25, N)

    common_exp = np.exp(-10 * (xx**2 / 2 + yy**2)) + \
        np.exp(-50 * ((xx - 0.5)**2 + (yy - 0.5)**2))
    u = 1 - 0.5 * common_exp
    v = 0.25 * common_exp
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    return u_hat, v_hat


def get_grid(L, N):
    """
    Generate an equidistant N x N square grid

    Returns
    -------
    xx, yy: array
        the N x N coordinates

    """
    x = (2 * L / N) * np.arange(-N / 2, N / 2)
    y = x
    xx, yy = np.meshgrid(x, y)
    return xx, yy


plt.close('all')
# X = np.random.rand(128, 128)
# X_hat = np.fft.fft2(X)
N_old = 128
X_hat, _ = initial_cond(N_old)
N = 256
Y_hat = up_scale(X_hat, N)
# Y_hat = down_scale(X_hat, N)
plt.figure()
ct = plt.contourf(np.fft.ifft2(X_hat), 100)
plt.colorbar(ct)
plt.figure()
ct = plt.contourf(np.fft.ifft2(Y_hat), 100)
plt.colorbar(ct)
