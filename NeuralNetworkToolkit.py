import numpy as np

class NeuralNetwork:

    def __init__(self, n_inputs, learn_rate=0.5, activation_function=0):
        self.layers = {}    # Weight/bias pairs stored under index in network
        self.nLayers = 0    # Excluding input layer
        self.nInputs = n_inputs
        self.nOutputs = n_inputs
        self.learnRate = learn_rate
        if activation_function != 0:
            self.activationFunc = activation_function
        else:
            self.activationFunc = self.sigmoid
        self.sq = np.vectorize(self.sq)

    def add_layer(self, n_nodes):
        self.nLayers += 1
        self.nOutputs = n_nodes
        if self.nLayers == 1:
            self.layers['1'] = (n_nodes, np.array(np.random.rand(n_nodes, self.nInputs)), np.random.rand(n_nodes).reshape(n_nodes, 1))
        else:
            self.layers[str(self.nLayers)] = (n_nodes, np.array(np.random.rand(n_nodes, self.layers[str(self.nLayers-1)][0])), np.random.rand(n_nodes).reshape(n_nodes, 1))

    def sq(self, x): return x**2

    def check_inputs(self, inp_arr):
        if inp_arr.shape != (long(self.nInputs), 1L):
            raise Exception("Incorrect inputs")

    def check_desired_output(self, des_out):
        if des_out.shape != (long(self.nOutputs), 1L):
            raise Exception("Incorrect Desired Output")

    def sigmoid(self, sig):
        sig = np.clip(sig, -500, 500)
        sig = 1.0/(1+np.exp(-sig))
        return sig

    def run(self, input_array):
        self.check_inputs(input_array)
        inp = input_array
        
        # Run forward pass
        for i in range(1, self.nLayers+1):
            n, weight_matrix, bias = self.layers[str(i)]
            raw_out = weight_matrix.dot(inp)
            raw_out = raw_out + bias
            out = self.activationFunc(raw_out)
            inp = out.copy()
        return inp.squeeze()

    def run_and_propagate(self, input_array, desired_output):
        self.check_inputs(input_array)
        self.check_desired_output(desired_output)
        inp = input_array
        prop_data = [] # Format (inp, out)
        
        # Run forward pass
        for i in range(1, self.nLayers+1):
            n, weight_matrix, bias = self.layers[str(i)]
            raw_out = weight_matrix.dot(inp)
            raw_out = raw_out + bias
            out = self.activationFunc(raw_out)
            prop_data.append((inp.copy(), out.copy()))
            inp = out.copy()

        des_out = desired_output
        derror_dprev_out = 0
        inpu, delta = 0, 0
        for i, x in enumerate(prop_data[::-1]):
            inpu, out = x
            inpu, out = np.array(inpu), np.array(out)
            layer_number = self.nLayers-i
            weight_matrix, bias = self.layers[str(layer_number)][1:]
            if i == 0:
                derror_dout = (out-des_out)
                dout_draw_out = out*(1-out)
                draw_out_dweights = inpu.T
                derror_dweights = (derror_dout * dout_draw_out) * draw_out_dweights
                derror_dprev_out = derror_dout*dout_draw_out
                delta = derror_dout * dout_draw_out
            else:
                dprev_out_dout = self.layers[str(layer_number+1)][1].T
                derror_dout = dprev_out_dout.dot(derror_dprev_out)
                dout_draw_out = out * (1-out)
                draw_out_dweights = inpu.T
                derror_dweights = (derror_dout * dout_draw_out) * draw_out_dweights
                delta = delta * (derror_dout)
            weight_matrix = weight_matrix - (self.learnRate * derror_dweights)
            bias = bias - (self.learnRate * delta)
            self.set_layer_data(layer_number, weight_matrix=weight_matrix, bias=bias)
        self.lastOut = inp
        return inp
        
    def set_layer_data(self, layer_number, weight_matrix=-1, bias=-1):
        if layer_number not in range(1, self.nLayers+1):
            raise Exception("Invalid layer number "+str(layer_number))
        cur_num, cur_mat, cur_bia = self.layers[str(layer_number)]
        if weight_matrix.shape != cur_mat.shape:
            raise Exception("Invalid Dimensions: "+weight_matrix.shape+" and "+self.layers[str(layer)].shape)
        cur_mat = weight_matrix
        if bias.shape != cur_bia.shape:
            raise Exception("Invalid Dimensions")
        cur_bia = bias
        self.layers[str(layer_number)] = (cur_num, weight_matrix, bias)#, cur_mat, cur_bia)

    def total_error(self, exp_array):
        diff = np.subtract(exp_array, self.lastOut)
        diff_sq = 0.5*self.sq(diff)
        absError = diff_sq.sum()
        return absError


# TEST

if __name__ == '__main__':
    N = NeuralNetwork(3, learn_rate=0.5)

    N.add_layer(2)
    N.add_layer(2)

    N.set_layer_data(1, weight_matrix=np.array([[-1,0,1],[1,1,0]]), bias=np.array([[0],[0]]))
    N.set_layer_data(2, weight_matrix=np.array([[-1,0],[1,1]]), bias=np.array([[0],[0]]))

    for i in range(1000):
        N.run_and_propagate(np.array([[1],[2],[3]]), np.array([[1],[0]]))
        
    print N.run_and_propagate(np.array([[1],[2],[3]]), np.array([[1],[0]]))
