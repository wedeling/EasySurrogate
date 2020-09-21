class Input_Layer:

    def __init__(self, n_neurons, bias, sequence_size):

        self.n_neurons = n_neurons
        self.bias = bias
        self.sequence_size = sequence_size
        self.h_history = []

        if self.bias:
            self.n_bias = 1
        else:
            self.n_bias = 0

    def compute_output(self, x_t):

        self.h_t = x_t

        self.h_history.append(self.h_t)
        if len(self.h_history) > self.sequence_size + 1:
            self.h_history.pop(0)

        return self.h_t

    # connect this layer to its neighbors
    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        self.layer_rm1 = layer_rm1
        self.layer_rp1 = layer_rp1

    def init_history(self):

        self.h_history = [None]
