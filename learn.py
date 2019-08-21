import numpy as np
import math
import sys

class Analysis() :
    def sigmoid(self,value):
        # activation function for hidden layer
        return 1/(1+np.exp(-value))

    def sigmoid_prime(self,value):
        s = self.sigmoid(value)
        return s*(1-s)

    def softmax(self,value) :
        # output activation that maps the final array into values that sums to 1 (probability distribution)
        if math.isnan(value[0][0]):
            sys.exit()
        e_x = np.exp(value-np.max(value))
        return e_x / np.sum(e_x)

    def softmax_prime(self,value):
        s = softmax(value)
        return np.diagflat(s) - np.dot(s,s.transpose())
    
    def cost_derivative(self,predicted_output,real_output):
        # returns the derivative of the cross-entropy loss function with respect to the ouput
        return predicted_output-real_output


    def forward_backward_propagation(self,first_input,real_output) :
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        activated_neurons = [first_input.reshape(784,1)]
        non_activated_neurons = []
        activated = activated_neurons[0]
        for i in range(self.number_of_hidden_layers):
            z = np.dot(self.weights[i],activated) + self.biases[i]
            non_activated_neurons.append(z)
            activated = self.sigmoid(z)
            activated_neurons.append(activated)
        
        z = np.dot(self.weights[-1],activated) + self.biases[-1]
        non_activated_neurons.append(z)
        activated = self.softmax(z)
        activated_neurons.append(activated)

            
        delta = self.cost_derivative(activated_neurons[-1],real_output)
        gradient_biases[-1] = delta
        gradient_weights[-1] = np.dot(delta,activated_neurons[-2].transpose())

        for i in range(2,len(self.weights)+1):
            non_activated_neuron = non_activated_neurons[-i]
            SP = self.sigmoid_prime(non_activated_neuron)
            delta = np.dot(self.weights[-i+1].transpose(),delta) * SP
            gradient_biases[-i] = delta
            gradient_weights[-i] = np.dot(delta,activated_neurons[-i-1].transpose())
       
        return gradient_biases,gradient_weights
    
    def testing(self,testing_data) :
        activated = testing_data.reshape(784,1)
        for weight,bias in zip(self.weights,self.biases):
            z = np.dot(weight,activated) + bias
            activated = self.sigmoid(z)
        
        return np.argmax(activated,0)


        

    
