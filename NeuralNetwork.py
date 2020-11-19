import scipy.special
import matplotlib.pyplot as mp
import numpy as np
class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes
        self.lr = learningrate
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
        
        self.activation_function = lambda x:scipy.special.expit(x)
        
        pass
    
    def train(self,input_list,target_list):
        
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih,inputs) 
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs  
        
        hidden_errors = np.dot(self.who.T,output_errors)
        # 改变权重举证
        self.who += self.lr * np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))        
        pass
    
    def query(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T 
        
        hidden_inputs = np.dot(self.wih,inputs) 
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.3
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
   
    train_data_file = open("d:/datasets/mnist_train.csv","r")
    train_data_list = train_data_file.readlines()
    train_data_file.close()
    i = 0
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
    
    
    test_data_file = open("d:/datasets/mnist_test.csv","r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    scores = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        # print(correct_label,"correct_lable")
        inputs = np.asfarray(all_values[1:])/255.0*0.99 + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        # print(label,"network's answer")
        if label == correct_label :
            scores.append(1)
        else:
            scores.append(0)
    pass
   
    # print(scores)
    scores_array = np.asarray(scores)
    print("performance =",scores_array.sum()/scores_array.size)
