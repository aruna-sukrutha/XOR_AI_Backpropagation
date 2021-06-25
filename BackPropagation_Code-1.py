import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
input = np.array([[0,0],[0,1],[1,0],[1,1]])
target_output = np.array([[0],[1],[1],[0]])
np.random.seed(10)
weights = np.random.uniform(size=(2,2)) #2 in input layer, 2 in hidden layer
bias = np.random.uniform(size=(1,2))#2 in hidden layer
output_weights = np.random.uniform(size=(2,1))#2 in hidden layer, 1 in output layer
output_bias = np.random.uniform(size=(1,1))#1 in output layer
for i in range(10000): #chosen 10000 for more learning rate,effective training and fast compilation
    y = np.dot(input,weights)
    y += bias
    x = sigmoid(y)
    z = np.dot(x,output_weights)
    z += output_bias
    actual_output = sigmoid(z)
    error = target_output - actual_output
    final = error * actual_output*(1-actual_output)#derivation
    error_hidden_layer = final.dot(output_weights.T)
    final_hidden_layer = error_hidden_layer * x*(1-x)#derivation
    output_weights += x.T.dot(final)
    output_bias += np.sum(final,axis=0,keepdims=True)
    weights += input.T.dot(final_hidden_layer)
    bias += np.sum(final_hidden_layer,axis=0,keepdims=True)
print("Input:\n",input)
print("Output of neural network:\n")
print(*actual_output)
