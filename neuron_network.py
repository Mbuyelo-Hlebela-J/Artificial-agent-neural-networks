import random
import neuron as ne
import equations as eqn
import json
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation="relu",output_activation="sigmoid"):
        # Initialize weights with small random values
        self.hidden_weights = [[random.uniform(-1,1)for _ in range(input_size)] for _ in range(hidden_size)]
        
        self.hidden_biases = [random.uniform(-1,1)for _ in range(hidden_size)] 
        #Ouput layer
        self.output_weights = [[0.1 for _ in range(hidden_size)] for _ in range(output_size)]
        self.output_biases = [0.0 for _ in range(output_size)]

        self.learning_rate = 0.1
        #Activation function selection
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation

        self.activations = {
            "sigmoid":(eqn.sigmoid,eqn.derivative_sigmoid),
            "tanh":(eqn.tanh,eqn.derivative_tanh),
            "relu" : (eqn.relu,eqn.derivative_relu),
            "leaky_relu":(eqn.leaky_relu,eqn.derivative_leaky_relu),
            "softmax":(eqn.softmax,None)
            }
        self.hidden_activation,self.hidden_derivative=self.activations[self.hidden_activation_name]
        self.output_activation,self.output_derivative=self.activations[self.output_activation_name]

        
    def neuron_forward(self, inputs, weights, bias,activation_func):
        z = eqn.dot_product(inputs, weights) + bias
        output = activation_func(z)

        return output, z

    def layer_forward(self, inputs, weights_matrix, biases,activation_func):
        zs = []
        for weights, bias in zip(weights_matrix,biases):
            z = eqn.dot_product(inputs,weights)+ bias
            zs.append(z)

        if activation_func.__name__== "softmax":
            outputs =eqn.softmax(zs)
        else:
            outputs= [activation_func(z) for z in zs]
        return outputs, zs

    def forward(self, inputs):
        self.hidden_outputs, self.hidden_zs = self.layer_forward(inputs, self.hidden_weights, self.hidden_biases,self.hidden_activation)
        self.final_outputs, self.final_zs = self.layer_forward(self.hidden_outputs, self.output_weights, self.output_biases,self.output_activation)
        return self.final_outputs[:]

    def train_batch(self, batch_inputs, batch_targets):
        # Gradient accumulators
        weight_gradients_hidden = [[0.0 for _ in row] for row in self.hidden_weights]
        bias_gradients_hidden = [0.0 for _ in self.hidden_biases]
        weight_gradients_output = [[0.0 for _ in row] for row in self.output_weights]
        bias_gradients_output = [0.0 for _ in self.output_biases]

        batch_size = len(batch_inputs)
        total_loss = 0


        for inputs, targets in zip(batch_inputs, batch_targets):
            self.forward(inputs)

            # Output layer gradients
            output_gradients = []
            for i in range(len(self.final_outputs)):
                error = self.final_outputs[i] - targets[i]
                dloss = 2 * error
                
                gradient = 0
                if(self.output_activation.__name__ == "softmax"):
                    gradient= self.final_outputs[i]-targets[i]
                else:
                    dz = self.output_derivative(self.final_zs[i])
                    gradient = dloss * dz
                    
                
                output_gradients.append(gradient)

                for j in range(len(self.output_weights[i])):
                    weight_gradients_output[i][j] += gradient * self.hidden_outputs[j]
                bias_gradients_output[i] += gradient

            # Hidden layer gradients
            hidden_gradients = []
            for j in range(len(self.hidden_outputs)):
                error = sum(output_gradients[a] * self.output_weights[a][j] for a in range(len(self.output_weights)))
                dz = self.hidden_derivative(self.hidden_zs[j])
                
                gradient = error * dz
                hidden_gradients.append(gradient)

                for k in range(len(self.hidden_weights[j])):
                    weight_gradients_hidden[j][k] += gradient * inputs[k]
                bias_gradients_hidden[j] += gradient

        # Apply average gradient updates
        for i in range(len(self.output_weights)):
            for j in range(len(self.output_weights[i])):
                self.output_weights[i][j] -= self.learning_rate * weight_gradients_output[i][j] / batch_size
            self.output_biases[i] -= self.learning_rate * bias_gradients_output[i] / batch_size

        for i in range(len(self.hidden_weights)):
            for j in range(len(self.hidden_weights[i])):
                self.hidden_weights[i][j] -= self.learning_rate * weight_gradients_hidden[i][j] / batch_size
            self.hidden_biases[i] -= self.learning_rate * bias_gradients_hidden[i] / batch_size
            
    def save_model(self, filename):
        model_data ={
            "hidden_weights:":self.hidden_weights,
            "hidden_biases:":self.hidden_biases,
            "output_weights:":self.output_weights,
            "output_biases:":self.output_biases,
            "hidden_activation_name:":self.hidden_activation_name,
            "output_activation_name:":self.output_activation_name
            }
        with open(filename,"w") as f:
            json.dump(model_data,f)
        print(f"Model saved to {filename}")
    def load_model(self,filename):
        with open(filename,"r") as f:
            model_data = json.load(f)
        self.hidden_weights = model_data["hidden_weights"]
        self.hidden_biases = model_data["hidden_biases"]
        self.output_weights = model_data["output_weights"]
        self.output_biases = model_data["output_biases"]
        
        self.hidden_activation_name = model_data["hidden_activation_name"]
        self.output_activation_name = model_data["output_activation_name"]

        
        self.hidden_activation,self.hidden_derivative=self.activations[self.hidden_activation_name]
        self.output_activation,self.output_derivative=self.activations[self.output_activation_name]

        print(f"Model loaded from {filename}")
        
    
        
    






        
