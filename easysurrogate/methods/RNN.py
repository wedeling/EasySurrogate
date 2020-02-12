import numpy as np

from .RNN_Layer import RNN_Layer

class RNN:
    
    def __init__(self, X, y, alpha = 0.001, decay_rate = 1.0, 
                 decay_step = 10**5, beta1 = 0.9, beta2 = 0.999, lamb = 0.0,
                 n_out = 1, param_specific_learn_rate = True, loss = 'squared', activation = 'tanh', 
                 activation_out = 'linear', n_softmax = 0, n_layers = 2, n_neurons = 16,
                 bias = True, sequence_size = 100, save = True, load=False, name='ANN', 
                 on_gpu = False, standardize_X = True, standardize_y = True, **kwargs):

        #number of input nodes
        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1
            
        #number of training samples
        self.n_train = X.shape[0]
        
        #use either numpy or cupy via xp based on the on_gpu flag
        global xp
        if on_gpu == False:
            import numpy as xp
        else:
            import cupy as xp
            
        self.on_gpu = on_gpu

        #standardize the training data
        if standardize_X == True:
            
            self.X_mean = xp.mean(X, axis = 0)
            self.X_std = xp.std(X, axis = 0)
            self.X = (X - self.X_mean)/self.X_std
        
        if standardize_y == True:
            self.y_mean = xp.mean(y, axis = 0)
            self.y_std = xp.std(y, axis = 0)
            self.y = (y - self.y_mean)/self.y_std
        
        #number of layers (hidden + output)
        self.n_layers = n_layers
        
        #number of layers in time
        self.sequence_size = sequence_size
        self.window_idx = 0
        self.max_windows_idx = self.n_train - sequence_size

        #number of neurons in a hidden layer
        self.n_neurons = n_neurons

        #number of output neurons
        self.n_out = n_out
        
        #use bias neurons
        self.bias = bias
        
        #number of bias neurons (0 or 1)
        self.n_bias = 0
        if self.bias:
            self.n_bias = 1
        
        #loss function type
        self.loss = loss

        #training rate
        self.alpha = alpha

        #L2 regularization parameter
        self.lamb = lamb

        #the rate of decay and decay step for alpha
        self.decay_rate = decay_rate        
        self.decay_step = decay_step

        #momentum parameter
        self.beta1 = beta1
        
        #squared gradient parameter
        self.beta2 = beta2
        
        #use parameter specific learning rate
        self.param_specific_learn_rate = param_specific_learn_rate

        #activation function of the hidden layers
        self.activation = activation
        
        #activation function of the output layer
        self.activation_out = activation_out
        
        #number of sofmax layers
        self.n_softmax = n_softmax
        
        #save the neural network after training
        self.save = save
        self.name = name
       
        self.loss_vals = []

        self.layers = []
        
        #input and hidden weight matrices
        self.W_in = []
        self.W_h = []
        
        #####################################
        # Initialize shared weight matrices #
        #####################################
        
        #input weights
        self.W_in.append(xp.random.randn(self.n_in + self.n_bias, self.n_neurons)*xp.sqrt(1.0/self.n_neurons))
        self.W_h.append(xp.random.randn(self.n_neurons + self.n_bias, self.n_neurons)*xp.sqrt(1.0/self.n_neurons))

        #hidden weights
        for r in range(1, self.n_layers):
            self.W_in.append(xp.random.randn(self.n_neurons + self.n_bias, self.n_neurons)*xp.sqrt(1.0/self.n_neurons))
            self.W_h.append(xp.random.randn(self.n_neurons + self.n_bias, self.n_neurons)*xp.sqrt(1.0/self.n_neurons))
            
        #output weights (no W_h matrix in the output layer)
        self.W_in.append(xp.random.randn(self.n_neurons + self.n_bias, self.n_out)*xp.sqrt(1.0/self.n_out))
        
        #########################
        # Create the RNN layers #
        #########################
        
        #add the hidden layers
        for r in range(self.n_layers):
            self.layers.append(RNN_Layer(self.W_in[r], self.W_h[r], 
                                         self.n_neurons, r, self.n_layers, 'tanh', 
                                         self.loss, self.bias, lamb = lamb,
                                         on_gpu=on_gpu))
        
        #add the output layer
        self.layers.append(RNN_Layer(self.W_in[self.n_layers], None, 
                                     self.n_out, r+1, self.n_layers, 'linear', 
                                     self.loss, False, lamb = lamb,
                                     on_gpu=on_gpu))
        
        self.connect_layers()
        
        self.print_network_info()
        
        
    #train the neural network        
    def train(self, n_batch, store_loss = True):
        
        #number of epochs
        self.n_epoch = 0
        
        for i in range(n_batch):

            #compute learning rate
            alpha = self.alpha*self.decay_rate**(np.int(i/self.decay_step))

            #run the batch
            self.batch(alpha=alpha, beta1=self.beta1, beta2=self.beta2)
          
            #store the loss value 
            if store_loss == True:
                l = 0.0
                for k in range(self.n_out):
                    l += self.layers[-1].L_t
                
                if np.mod(i, 1000) == 0:
                    loss_i = xp.mean(l)
                    print('Batch', i, 'learning rate', alpha ,'loss:', loss_i)
                    #note: appending a cupy value to a list is inefficient - if done every iteration
                    #it will slow down executing significantly
                    self.loss_vals.append(loss_i)
                    
        # if self.save == True:
        #     self.save_ANN()
                    
        
    #run the network forward
    def feed_forward(self, back_prop = False):
        
        #generates the next sequence of continuous indices for the inputs
        idx = self.slide_window(self.sequence_size)
        x_sequence = self.X[idx, :]
        y_sequence = self.y[idx, :]
        
        #predicted output sequence        
        y_hat_t = xp.zeros([self.sequence_size, self.n_out])
        
        if back_prop:
            #zero out the gradients of the previous sequence
            for r in range(self.n_layers+1):
                self.layers[r].L_grad_W_in = 0.0
                self.layers[r].L_grad_W_h = 0.0
            #set loss to zero
            self.layers[-1].L_t = 0.0
        
        #loop over all inputs of the input sequence
        for t in range(self.sequence_size):
        
            #current input
            x_t = x_sequence[t].reshape([self.n_in, 1])
            
            #if a bias neuron is used, add 1 to the end of x_t
            if self.bias == True:
                x_t = xp.vstack([x_t, xp.ones(1)])

            input_feat = x_t
            for r in range(self.n_layers+1):
                #compute the output, i.e. the hidden state, of the current layer
                hidden_state = self.layers[r].compute_output(input_feat)
                #hidden state becomes input for the next layer 
                input_feat = hidden_state
            
            #final hidden state is the output
            y_hat_t[t, :] = hidden_state
            
            if back_prop:
                self.back_prop(y_sequence[t].reshape([self.n_out, 1]))
            
        return y_hat_t
    
    def slide_window(self, n):
        
        if self.window_idx > self.max_windows_idx:
            self.window_idx = 0
            self.n_epoch += 1
            print("Performed", self.n_epoch, "epochs.")
            
        idxs = range(self.window_idx, self.window_idx + self.sequence_size)
        self.window_idx += n
        
        return idxs
    
    #back propagation algorithm    
    def back_prop(self, y_t):

        #start back propagation over hidden layers, starting with output layer
        for r in range(self.n_layers, 0, -1):
            self.layers[r].back_prop(y_t)

    #update step of the weights
    def batch(self, alpha=0.001, beta1=0.9, beta2=0.999, t=0):
        
        self.feed_forward(back_prop = True)
        
        for r in range(self.n_layers+1):

            layer_r = self.layers[r]
            
            #moving average of gradient and squared gradient magnitude
            layer_r.V_in = beta1*layer_r.V_in + (1.0 - beta1)*layer_r.L_grad_W_in
            layer_r.A_in = beta2*layer_r.A_in + (1.0 - beta2)*layer_r.L_grad_W_in**2

            #TODO: find better way, this if condition appears three times
            if r < self.n_layers:
                layer_r.V_h = beta1*layer_r.V_h + (1.0 - beta1)*layer_r.L_grad_W_h
                layer_r.A_h = beta2*layer_r.A_h + (1.0 - beta2)*layer_r.L_grad_W_h**2

            #select learning rate            
            if self.param_specific_learn_rate == False:
                #same alpha for all weights
                alpha_in = alpha
                alpha_h = alpha
            #param specific learning rate
            else:
                #RMSProp
                alpha_in = alpha/(xp.sqrt(layer_r.A_in + 1e-8))
                if r < self.n_layers:
                    alpha_h = alpha/(xp.sqrt(layer_r.A_h + 1e-8))
                
                #Adam
                #alpha_t = alpha*xp.sqrt(1.0 - beta2**t)/(1.0 - beta1**t)
                #alpha_i = alpha_t/(xp.sqrt(layer_r.A + 1e-8))

            #gradient descent update step
            if self.lamb > 0.0:
                #with L2 regularization
                layer_r.W = (1.0 - layer_r.Lamb*alpha_i)*layer_r.W - alpha_i*layer_r.V
            else:
                #without regularization
                layer_r.W_in = layer_r.W_in - alpha_in*layer_r.V_in
                if r < self.n_layers:
                    layer_r.W_h = layer_r.W_h - alpha_h*layer_r.V_h
           
                #Nesterov momentum
                # layer_r.W += -alpha*beta1*layer_r.V

           
    #connect each layer in the NN with its previous and the next      
    def connect_layers(self):
        
        self.layers[0].meet_the_neighbors(None, self.layers[1])
        self.layers[-1].meet_the_neighbors(self.layers[-2], None)
        
        for i in range(1, self.n_layers):
            self.layers[i].meet_the_neighbors(self.layers[i-1], self.layers[i+1])
            
    #return the number of weights
    def get_n_weights(self):
        
        n_weights = 0
        
        #sum weight matrix sizes of all hidden layers (sizes W_in and W_h)
        for i in range(self.n_layers):
            n_weights += self.W_in[i].size + self.W_h[i].size
            
        #sum weight matrix sizes of output layers (size W_in only)
        n_weights += self.W_in[i].size
        
        print('This neural network has', n_weights, 'weights.')

        return n_weights
            
    def print_network_info(self):
        print('===============================')
        print('RNN parameters')
        print('===============================')
        print('Number of layers =', self.n_layers)
        print('Number of features =', self.n_in)
        print('Loss function =', self.loss)
        print('Number of neurons per hidden layer =', self.n_neurons)
        print('Number of output neurons =', self.n_out)
        print('Activation hidden layers =', self.activation)
        print('Activation output layer =', self.activation_out)
        print('On GPU =', self.on_gpu)
        self.get_n_weights()
        print('===============================')
