import equations as eqn


class Neuron:

    #using the sigmoid fustion
    def neuron_forward(self,inputs,weights,bias):
        z = eqn.dot_product(inputs,weights) + bias
        output = eqn.sigmoid(z)
        return output,z
    #Layers
    def layer_forward(self,inputs,weights_matrix, biases):
        outputs = []
        zs = []
        for weights,bias in zip(weights_matrix,biases):
            output, z = self.neuron_forward(inputs,weights,bias)
            outputs.append(output)
            zs.append(z)
        return outputs,zs
    #Train layer
    def train_layer(inputs,targets,weights_matrix,biases,learning_rate):
        layer_outputs = []
        zs = []

        #---forward pass for each neuron---
        for weights, bias in zip(weights_matrix,biases):
            output,z = self.neuron_forward(inputs,weights,bias)
            layer_outputs.append(output)
            zs.append(z)
        #---backpropagation---

        new_weights_matrix = []
        new_biases = []

        for i in range(len(weights_matrix)):
            output = layer_outputs[i]
            z = zs[i]
            target = targets[i]

            error = output - target
            dloss_doutput = 2*error
            doutput_dz = eqn.derivative_sigmoid(z)
            dloss_dz = dloss_doutput*doutput_dz

            #weight gradients
            gradients_w = [dloss_dz*x for x in inputs]
            gradient_b = dloss_dz

            #update weights and bias
            new_weights =[
                w - learning_rate*gw
                for w, gw in zip(weights_matrix[i],gradients_w)
                ]
            new_bias = biases[i] - learning_rate*gradient_b
            new_weights_matrix.append(new_weights)
            new_biases.append(new_bias)
        return new_weights_matrix,new_biases,layer_outputs













            
