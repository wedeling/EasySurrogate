"""
Deep Active Subspace layer
"""
import numpy as np

from .Layer import Layer


class DAS_Layer(Layer):
    """
    Deep Active Subspace layer with weights parameterized by Gram-Schmidt
    orthogonalization. Place between input layer and first hidden layer.
    Inherits from the standard Layer class.

    Method:
        Tripathy, Rohit, and Ilias Bilionis. "Deep active subspaces: A scalable
        method for high-dimensional uncertainty propagation."
        ASME 2019 International Design Engineering Technical Conferences and
        Computers and Information in Engineering Conference.
        American Society of Mechanical Engineers Digital Collection, 2019.
    """

    def __init__(self, d, n_layers, bias, batch_size=1):
        """
        Initialize the DAS_layer oject.

        Parameters
        ----------
        d : int
            The dimension of the active subspace.
        bias : boolean, optional
            Add a bias neuron to the DAS layer. NOT IMPLEMENTED YET.
            The default is False.
        batch_size : int, optional
            The batch size. The default is 1.

        Returns
        -------
        None.

        """
        super().__init__(d, 1, n_layers, 'linear', 'none',
                         bias=bias, batch_size=batch_size)
        self.d = d
        self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
        self.name = 'DAS_layer'

    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        """
        Connect this layer to its neighbors

        Parameters
        ----------
        layer_rm1 : Layer object or None
            The layer before at index r - 1.
        layer_rp1 : Layer object or None
            The layer after at index r + 1.

        Returns
        -------
        None.

        """
        super().meet_the_neighbors(layer_rm1, layer_rp1)
        self.D = layer_rm1.n_neurons

    def init_weights(self):
        """
        Initialize the weights and othe related matrices of this DAS layer.

        Returns
        -------
        None.

        """
        # initialize the weights of the layer, which parameterize the
        # Gram-Schmidt vectors w
        self.Q = np.random.randn(self.layer_rm1.n_neurons, self.n_neurons) * \
            np.sqrt(1.0 / self.layer_rm1.n_neurons)
        # the unnormalized Gram-Schmidt vectors
        self.w = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        # compute the Gram-Schmidt vectors, given Q
        self.compute_weights()
        # the gradient of the loss function
        self.L_grad_W = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        # momentum
        self.V = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        # squared gradient
        self.A = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])

    def compute_weights(self):
        """
        Compute the Gram-Schmidt vectors w(Q)

        Returns
        -------
        None.

        """
        # This is a stadard Gram-Schmidt implementation, computing orthonormal
        # vectors from another set of vectors Q = [q1 q2 ... q_d].
        self.w[:, 0] = self.Q[:, 0]

        for i in range(1, self.d):
            self.w[:, i] = self.Q[:, i]
            for j in range(i):
                self.w[:, i] -= (np.dot(self.w[:, j], self.Q[:, i]) /
                                 np.dot(self.w[:, j], self.w[:, j])) * self.w[:, j]

        # the vectors w are not normalized, the weights of the neural network (W)
        # are normalized as w / norm(w)
        self.norm_w = np.linalg.norm(self.w, axis=0)
        self.W = self.w / self.norm_w

    def compute_L_grad_Q(self):
        """
        Compute the gradient of the loss function wrt the weights of this layer.
        Unlike standard layer, the weights are parameterized by Q, and so we
        must compute the gradient of the loss function with respect to Q here.

        Returns
        -------
        None.

        """
        # This part is standard back prop: compute L_grad_W
        h_rm1 = self.layer_rm1.h
        delta_ho_grad_Phi = self.delta_ho * self.grad_Phi
        self.L_grad_W = np.dot(h_rm1, delta_ho_grad_Phi.T)
        # here we compute W_grad_q_ij, the gardient of Q wrt each entry in Q
        # The results are stored in a 3D array of shape (Dd, D, d), which contains
        # the Dd matrices dW / dq_ij, i=1,...,D, j=1,...,d.
        self.W_grad_q_ij = self.compute_W_grad_q_ij()
        # This computes the gradient of loss function wrt the matrix Q via the
        # chain rule: dL / dQ = dL / dW * dW / dq_11 + ... + dl / dW * dW / dq_Dd
        self.L_grad_Q = np.sum(self.W_grad_q_ij * self.L_grad_W,
                               axis=(1, 2)).reshape([self.D, self.d])

    def compute_y_grad_Q(self):
        """
        Compute the gradient of the output wrt the weights of this layer.
        Unlike standard layer, the weights are parameterized by Q, and so we
        must compute the gradient of the loss function with respect to Q here.

        Returns
        -------
        None.

        """
        # here we compute W_grad_q_ij, the gardient of Q wrt each entry in Q
        # The results are stored in a 3D array of shape (Dd, D, d), which contains
        # the Dd matrices dW / dq_ij, i=1,...,D, j=1,...,d.
        self.W_grad_q_ij = self.compute_W_grad_q_ij()
        # This computes the gradient of loss function wrt the matrix Q via the
        # chain rule: dL / dQ = dL / dW * dW / dq_11 + ... + dl / dW * dW / dq_Dd
        self.y_grad_Q = np.sum(self.W_grad_q_ij * self.y_grad_W,
                               axis=(1, 2)).reshape([self.D, self.d])

    def compute_W_grad_q_ij(self):
        """
        Compute the gradient of the weights W (the normalized Gram-Schmidt vectors)
        with respect to every entry of the matrix Q.

        Returns
        -------
        dWdq_ij : array
            The Dd gradients dW / dq_ij, i=1,...,D, j=1,...,d. Stored in an array
            of shape (Dd, D, d).

        """

        d = self.d
        D = self.D
        # create a list of the column vectors of W and Q, with shape (D,1)
        w = [w_i.reshape([self.D, 1]) for w_i in list(self.w.T)]
        q = [q_i.reshape([self.D, 1]) for q_i in list(self.Q.T)]
        # gradients of the Gram-Schmidt vectors dw_i / dq_k
        grads = np.zeros([d**2, D, D])
        # gradients of the normed Gram-Schmidt vectos d(w_i / ||w_i||_2) / dq_k
        grads_normed = np.zeros([d**2, D, D])
        # D_ij matrices
        D_ij = np.zeros([d**2, D, D])
        # for w_1, the gradient and D_ij matrices have a simple form. Compute these
        # outside loop.
        I_D = np.eye(D)
        D_ij[0] = I_D
        grads[0] = I_D
        grads_normed[0] = I_D / self.norm_w[0] - np.dot(w[0], w[0].T) / self.norm_w[0]**3

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
            norm_mat = I_D / self.norm_w[i] - np.dot(w[i], w[i].T) / self.norm_w[i]**3
            for k in range(d):
                # convert 2D index (i,j) to scalar index idx
                idx = np.ravel_multi_index([i, k], dims=(d, d))
                # due to GS, the gradient dw_i / dq_k is zero for k > i, so compute if i >= k
                if i >= k:
                    # a 'normal' gradient dw_i / dq_i
                    if i == k:
                        # index of dw_{i-1} / dq_{i-1}
                        idx1 = np.ravel_multi_index([i - 1, i - 1], dims=(d, d))
                        grad_ik = grads[idx1] - \
                            np.dot(w[i - 1], w[i - 1].T) / np.dot(w[i - 1].T, w[i - 1])
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

        return dWdq_ij

    def back_prop(self, y_i=None, jacobian=False):
        """
        Perform the backpropogation operations of the current layer. Unlike in the
        standard Layer, we compute the gradient with respect to the matrix Q, since
        the weights are parameterized by Q.

        Returns
        -------
        None.

        """
        # compute the standard gradient of the loss function wrt the activation
        # functions, uses method from parent class.
        self.compute_delta_ho()
        # compute the loss gradient wrt Q, since we must update Q
        self.compute_L_grad_Q()

        if jacobian:
            self.compute_delta_hy()
            self.compute_y_grad_W()
            self.compute_y_grad_Q()
            self.L_grad_Q = -self.y_grad_Q


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
            np.dot(w_j.T, q_i) * np.eye(q_i.size)) / np.dot(w_j.T, w_j)
