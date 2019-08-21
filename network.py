import numpy as np
from learn import Analysis
import random

class Network (Analysis):
    def __init__(self,sizes):
        self.number_of_hidden_layers = len(sizes) - 2
        self.sizes = sizes
        self.biases = [np.random.randn(rows,1) for rows in sizes[1:]]
        self.weights = [np.random.randn(columns,rows) for rows,columns in zip(sizes[:-1],sizes[1:])]

    def stochastic_gradient_descent(self,epoch,mini_batch_size,learning_rate,training_data,testing_data):
        training_data = training_data
        for i in range(epoch) :
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0,len(training_data),mini_batch_size)
            ]
            count = 0
            for mini_batch in mini_batches :
                self.update(mini_batch,learning_rate)
            print('Epoch {0} finished'.format(i))
            self.test(testing_data)
        

    def update(self,mini_batch,learning_rate) :
        grad_biases = [np.zeros(b.shape) for b in self.biases]
        grad_weights = [np.zeros(w.shape) for w in self.weights]

        for (data, label) in mini_batch :
            var_gradient_biases, var_gradient_weights = self.forward_backward_propagation(data,label)
            grad_biases = [gb+vgb for gb,vgb in zip(grad_biases,var_gradient_biases)]
            grad_weights = [gw+vgw for gw,vgw in zip(grad_weights,var_gradient_weights)]
        

        
        self.biases = [b - (learning_rate/len(mini_batch))*gb for b,gb in zip(self.biases,grad_biases)]
        self.weights = [w - (learning_rate/len(mini_batch))*gw for w,gw in zip(self.weights,grad_weights)]
    
    def test(self,testing_data) :
        test = testing_data
        result = [(self.testing(data),np.argmax(label)) for (data,label) in test]
        print(str(sum(int(x==y) for x,y in result) / 10000 * 100) + "%" + " accurate \n")
        



    def __str__(self):
        return 'This network as %s layers.' % (self.number_of_layers)








        