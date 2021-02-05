import numpy as np

class DAS_Layer:
    
    def __init__(self, d, bias = False, batch_size=1):
        self.n_neurons = d
        self.d = d
        self.batch_size = batch_size
        self.r = 1
        if bias:
            self.n_bias = 1
        else:
            self.n_bias = 0
        self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
        self.name = 'DAS_layer'

    # connect this layer to its neighbors
    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        self.layer_rm1 = layer_rm1
        self.layer_rp1 = layer_rp1
        self.D = layer_rm1.n_neurons
        self.seed_neurons()
    
    # initialize the neurons of this layer
    def seed_neurons(self):

        self.Q = np.random.randn(self.layer_rm1.n_neurons, self.n_neurons) * \
            np.sqrt(1.0 / self.layer_rm1.n_neurons)
        self.w = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        self.compute_weights()
        self.L_grad_W = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        self.V = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        self.A = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
    
    def compute_weights(self):
        self.w[:, 0] = self.Q[:, 0]

        for i in range(1, self.d):
            self.w[:, i] = self.Q[:, i]
            for j in range(i):
                self.w[:, i] -= (np.dot(self.w[:, j], self.Q[:, i]) / 
                                 np.dot(self.w[:, j], self.w[:, j])) * self.w[:, j]

        self.norm_w = np.linalg.norm(self.w, axis=0)
        self.W = self.w / self.norm_w

    # compute the output of the current layer in one shot using matrix -
    # vector/matrix multiplication
    def compute_output(self, batch_size):

        a = np.dot(self.W.T, self.layer_rm1.h)
        self.h = a
    
    # compute the gradient of the loss function wrt the activation functions of this layer
    def compute_delta_ho(self):
        # get the delta_ho values of the next layer (layer r+1)
        delta_ho_rp1 = self.layer_rp1.delta_ho
        # get the grad_Phi values of the next layer
        grad_Phi_rp1 = self.layer_rp1.grad_Phi
        # the weight matrix of the next layer
        W_rp1 = self.layer_rp1.W

        self.delta_ho = np.dot(W_rp1, delta_ho_rp1 * grad_Phi_rp1)[0:self.n_neurons, :]
        
    # compute the gradient of the loss function wrt the weights of this layer
    def compute_L_grad_Q(self):
        h_rm1 = self.layer_rm1.h
        delta_ho_grad_Phi = self.delta_ho * self.grad_Phi
        self.L_grad_W = np.dot(h_rm1, delta_ho_grad_Phi.T)
        self.W_grad_q_ij = self.compute_W_grad_q_ij()
        self.L_grad_Q = np.sum(self.W_grad_q_ij * self.L_grad_W, axis=(1, 2)).reshape([self.D, self.d])
        
    def compute_W_grad_q_ij(self):

        d = self.d; D = self.D
        #create a list of the column vectors of W and Q, with shape (D,1)
        w = [w_i.reshape([self.D, 1]) for w_i in list(self.w.T)]
        q = [q_i.reshape([self.D, 1]) for q_i in list(self.Q.T)]
        #gradients of the Gram-Schmidt vectors dw_i / dq_k
        grads = np.zeros([d**2, D, D])
        #gradients of the normed Gram-Schmidt vectos d(w_i / ||w_i||_2) / dq_k
        grads_normed = np.zeros([d**2, D, D])
        #D_ij matrices
        D_ij = np.zeros([d**2, D, D])
        #for w_1, the gradient and D_ij matrices have a simple form. Compute these 
        #outside loop.
        I_D = np.eye(D)
        D_ij[0] = I_D
        grads[0] = I_D
        grads_normed[0] = I_D/self.norm_w[0] - np.dot(w[0], w[0].T)/self.norm_w[0]**3

        #the D_ij matrices must be precomputed
        #loop over dw_i, i = 2,...,d
        for i in range(1, d):
            #loop over all dq_j, j = 1,...,d
            for j in range(d):
                #convert 2D index (i,j) to scalar index idx
                idx = np.ravel_multi_index([i, j], dims=(d,d))
                if i >= j:
                    D_ij[idx] = compute_D_ij(w[j], q[i])           
        
        #construct the d^2 gradient matrices dw_i / dq_k
        for i in range(1, d):
            #the matrix by which the gradient d_wi / dq_k must be premultiplied
            norm_mat = I_D/self.norm_w[i] - np.dot(w[i], w[i].T)/self.norm_w[i]**3
            for k in range(d):
                #convert 2D index (i,j) to scalar index idx
                idx = np.ravel_multi_index([i, k], dims=(d,d))
                #due to GS, the gradient dw_i / dq_k is zero for k > i, so compute if i >= k
                if i >= k:
                    #a 'normal' gradient dw_i / dq_i
                    if i == k:
                        # index of dw_{i-1} / dq_{i-1}
                        idx1 = np.ravel_multi_index([i-1, i-1], dims = (d,d))
                        grad_ik = grads[idx1] - np.dot(w[i-1], w[i-1].T)/np.dot(w[i-1].T, w[i-1])
                    #a 'shear' gradient dw_i / dq_k where i is not k
                    else:
                        grad_ik = 0.0
                        #loop over j=1,...,i - 1
                        for j in range(i):
                            #index of D_{ij}
                            idx2 = np.ravel_multi_index([i, j], dims = (d,d))
                            #index of dw_j / dq_k
                            idx3 = np.ravel_multi_index([j, k], dims = (d,d))
                            grad_ik -= np.dot(D_ij[idx2], grads[idx3])
                    #store dw_i / dq_k
                    grads[idx] = grad_ik
                    #compute the derivates of w_i / ||w_i||_2
                    grads_normed[idx] = np.dot(norm_mat, grad_ik)
                    
        #re-order the gradient matrices to get get dW / dq_ij, i=1,...,D, j = 1,...,d.
        #Here, W = [w_1, w_2, ..., w_d] is the matrix of all orthogonal vectors
        dWdq_ij = np.zeros([D*d, D, d])
        counter = 0
        for i in range(D):
            for j in range(d):
                #selects d dw_k / dq_j matrices (k = 1,...,d and fixed j)
                idx = np.arange(j, d**2, d)
                #the gradient of W wrt scalar q_ij: select the i-th column of 
                #the d dw_k / dq_j matrices (k = 1,...,d and fixed j)
                dWdq_ij[counter] = grads_normed[idx, :, i].T
                counter += 1
        
        return dWdq_ij

    # perform the backpropogation operations of the current layer
    def back_prop(self, y_i):

        self.compute_delta_ho()
        self.compute_L_grad_Q()

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
            (2.0*np.dot(w_j.T, q_i) / np.dot(w_j.T, w_j)) * np.dot(w_j, w_j.T) + 
            np.dot(w_j.T, q_i) * np.eye(q_i.size)) / np.dot(w_j.T, w_j)   