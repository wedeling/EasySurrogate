import numpy as np


def compute_D_ij(w_j, q_i):
    """
    Compute the D_ij matrices which appear in the gradients of the
    Gram-Schmidt vectors w_i wrt the original unnormalized vectors q_j.
    These matrices are only defined for i unequal to j.

    D_ij = 1/(w_j^Tw_j) [w_jq_i^T - (2w_j^Tq_i)/(w_j^Tw_j)*w_jw_j^T + w_j^Tq_i*I]
    where I is the D x D identity matrix.

    Parameters
    ----------
    w_j : array, shape (D,1)
        The orthogonal, but not yet normalized, vector w_i, i=1,...,d.
    q_i : array, shape (D,1)
        An orginal, non-orthonogal, vector q_j, j=1,...,d.

    Returns
    -------
    D_ij : array
        The D x D matrix described above.

    """
    return (np.dot(w_j, q_i.T) -
            (2.0 * np.dot(w_j.T, q_i) / np.dot(w_j.T, w_j)) * np.dot(w_j, w_j.T) +
            np.dot(w_j.T, q_i) * np.eye(D)) / np.dot(w_j.T, w_j)


# dimension full-scale problem
D = 5

# random, non-orthonormal vectors
q1 = np.array([0.1, 0.34, 1.9, 2.4, 3.1]).reshape([D, 1])
q2 = np.array([0.4, 0.44, 0.78, 1.9, 1.1]).reshape([D, 1])
q3 = np.array([1.0, 1.14, 0.2, 1.69, 0.26]).reshape([D, 1])
q4 = np.array([1.22, 1.04, 0.9, 0.69, 0.61]).reshape([D, 1])
q = [q1, q2, q3, q4]

# dimension active subspace
d = 4

# gram-schmidt
w1 = q1
w2 = q2 - np.dot(w1.T, q2) / np.dot(w1.T, w1) * w1
w3 = q3 - np.dot(w1.T, q3) / np.dot(w1.T, w1) * w1 - np.dot(w2.T, q3) / np.dot(w2.T, w2) * w2
w4 = q4 - np.dot(w1.T, q4) / np.dot(w1.T, w1) * w1 - np.dot(w2.T, q4) / np.dot(w2.T, w2) * w2 - \
    np.dot(w3.T, q4) / np.dot(w3.T, w3) * w3
w = [w1, w2, w3, w4]

# compute norms
norm_w1 = np.linalg.norm(w1)
norm_w2 = np.linalg.norm(w2)
norm_w3 = np.linalg.norm(w3)
norm_w4 = np.linalg.norm(w4)
norm_w = [norm_w1, norm_w2, norm_w3, norm_w4]

# gradients
grads = np.zeros([d**2, D, D])
# normed gradients
grads_normed = np.zeros([d**2, D, D])
# D_ij matrices
D_ij = np.zeros([d**2, D, D])
# for w_1, the gradient and D_ij matrices have a simple form. Compute these
# outside loop.
I_D = np.eye(D)
D_ij[0] = I_D
grads[0] = I_D
grads_normed[0] = I_D / norm_w[0] - np.dot(w[0], w[0].T) / norm_w[0]**3

# the D_ij matrices must be precomputed
# loop over dw_i, i = 2,...,d
for i in range(1, d):
    # loop over all dq_j, j = 1,...,d
    for j in range(d):
        # convert 2D index (i,j) to scalar index idx
        idx = np.ravel_multi_index([i, j], dims=(d, d))
        if i >= j:
            D_ij[idx] = compute_D_ij(w[j], q[i])

# construct the d^2 gradient matrices dw_i / dq_k
for i in range(1, d):
    # the matrix by which the gradient d_wi / dq_k must be premultiplied
    norm_mat = I_D / norm_w[i] - np.dot(w[i], w[i].T) / norm_w[i]**3
    for k in range(d):
        # convert 2D index (i,j) to scalar index idx
        idx = np.ravel_multi_index([i, k], dims=(d, d))
        # due to GS, the gradient dw_i / dq_k is zero for k > i, so compute if i >= k
        if i >= k:
            # a 'normal' gradient dw_i / dq_i
            if i == k:
                # index of dw_{i-1} / dq_{i-1}
                idx1 = np.ravel_multi_index([i - 1, i - 1], dims=(d, d))
                grad_ik = grads[idx1] - np.dot(w[i - 1], w[i - 1].T) / np.dot(w[i - 1].T, w[i - 1])
            # a 'shear' gradient dw_i / dq_k where i is not k
            else:
                grad_ik = 0.0
                # loop over j=1,...,i - 1
                for j in range(i):
                    # index of D_{ij}
                    idx2 = np.ravel_multi_index([i, j], dims=(d, d))
                    # index of dw_j / dq_k
                    idx3 = np.ravel_multi_index([j, k], dims=(d, d))
                    grad_ik -= np.dot(D_ij[idx2], grads[idx3])
            # store dw_i / dq_k
            grads[idx] = grad_ik
            # compute the derivates of w_i / ||w_i||_2
            grads_normed[idx] = np.dot(norm_mat, grad_ik)

# re-order the gradient matrices to get get dW / dq_ij, i=1,...,D, j = 1,...,d.
# Here, W = [w_1, w_2, ..., w_d] is the matrix of all orthogonal vectors
dWdq_ij = np.zeros([D * d, D, d])
counter = 0
for i in range(D):
    for j in range(d):
        # selects d dw_k / dq_j matrices (k = 1,...,d and fixed j)
        idx = np.arange(j, d**2, d)
        # the gradient of W wrt scalar q_ij: select the i-th column of
        # the d dw_k / dq_j matrices (k = 1,...,d and fixed j)
        dWdq_ij[counter] = grads_normed[idx, :, i].T
        counter += 1
