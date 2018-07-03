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
            self.layers['1'] = (n_nodes, np.random.rand(self.nInputs, n_nodes), np.random.rand(1,n_nodes))
        else:
            self.layers[str(self.nLayers)] = (n_nodes, np.matrix(np.random.rand(self.layers[str(self.nLayers-1)][0], n_nodes)), np.random.rand(1, n_nodes))

    def sq(self, x): return x**2

    def check_inputs(self, inp_arr):
        if inp_arr.shape != (long(1), long(self.nInputs)): raise Exception("Inncorrect inputs")

    def check_desired_output(self, des_out):
        if des_out.shape != (long(1), long(self.nInputs)): raise Exception("Inncorrect inputs")

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
            raw_out = np.dot(inp, weight_matrix.T)
            raw_out = raw_out + bias
            out = self.activationFunc(raw_out)
            inp = out.copy()
        return inp

    def run_and_propagate(self, input_array, desired_output):
        self.check_inputs(input_array)
        self.check_desired_output(desired_output)
        inp = input_array
        prop_data = [] # Format [inp, out]

        # Run forward pass
        for i in range(1, self.nLayers+1):
            n, weight_matrix, bias = self.layers[str(i)]
            raw_out = np.dot(inp, weight_matrix.T)
            raw_out = raw_out + bias
            out = self.activationFunc(raw_out)
            prop_data.append((inp.copy(), out.copy()))
            inp = out.copy()

        # Back Propagate
        des_out = desired_output
        derror_dprev_out = 0
        inpu, delta = 0, 0
        for i, x in enumerate(prop_data[::-1]):
            inpu, out = x   # inpu = input into layer, out = output of layer
            layer_number = self.nLayers-i
            weight_matrix, bias = self.layers[str(layer_number)][1:]
            if i == 0:
                derror_dout = out-des_out
                dout_draw_out = np.multiply(out, 1-out)
                draw_out_dweights = inpu
                derror_dweights = derror_dout*dout_draw_out*draw_out_dweights
                derror_dprev_out = derror_dout * dout_draw_out
                delta = derror_dout * dout_draw_out
            else:
                dprev_out_dout = self.layers[str(layer_number+1)][1].T
                dout_draw_out = np.multiply(out, 1-out)
                draw_out_dweights = inpu
                derror_dweights = derror_dprev_out * np.dot(dout_draw_out, dprev_out_dout) * draw_out_dweights 
                derror_dprev_out = np.dot(derror_dprev_out, dprev_out_dout)
                delta = np.dot(delta, dprev_out_dout) * dout_draw_out
            weight_matrix = weight_matrix - self.learnRate*derror_dweights.T
            bias = bias - self.learnRate*delta
            self.set_layer_data(layer_number, weight_matrix = weight_matrix, bias = bias)
        return inp
        
    def set_layer_data(self, layer_number, weight_matrix=-1, bias=-1):
        if layer_number not in range(1, self.nLayers+1):
            raise Exception("Invalid layer number "+str(layer_number))
        cur_num, cur_mat, cur_bia = self.layers[str(layer_number)]
        if type(weight_matrix) == np.ndarray:
            if weight_matrix.shape != self.layers[str(layer_number)][1].shape:
                raise Exception("Invalid Dimensions: "+weight_matrix.shape+" and "+self.layers[str(layer)].shape)
            else:
                cur_mat = weight_matrix
        if type(bias) == np.ndarray:
            if bias.shape != self.layers[str(layer_number)][2].shape:
                raise Exception("Invalid Dimensions")
            else:
                cur_bia = bias
        self.layers[str(layer_number)] = (cur_num, cur_mat, cur_bia)

    def total_error(self, exp_array):
        diff = np.subtract(exp_array, self.lastOut)
        diff_sq = self.sq(diff)
        absError = diff_sq.sum()
        return diff_sq, absError

# TEST

if __name__ == '__main__':
    N = NeuralNetwork(2, learn_rate=0.5)
    N.add_layer(2)
    N.add_layer(2)
    N.set_layer_data(1, weight_matrix = np.array([[0.15,0.20],[0.25,0.30]]), bias= np.array([[0.35, 0.35]]))
    N.set_layer_data(2, weight_matrix = np.array([[0.40,0.45],[0.5,0.55]]), bias = np.array([[0.6, 0.6]]))

    for i in range(10000):
        N.run_and_propagate(np.array([[0.05, 0.10]]), np.array([[0.01, 0.99]]))
    print N.run_and_propagate(np.array([[0.05, 0.10]]), np.array([[0.01, 0.99]]))
