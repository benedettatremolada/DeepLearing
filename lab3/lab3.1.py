import numpy as np
import tensorflow as tf

tf.random.set_seed(0)


#create model
''' generic function MLP with no fixed layers or neurons (weights and bias must be LIST!)
def MLP(x, weights, biases):
    if (len(weights)!=len(biases)): return 'weights and biases must have the same size'

    for i in range(len(weights)-1): 
        x=tf.math.sigmoid(tf.add(tf.matmul(input,weights[i]), biases[i])) #middle layers with a sigmoid as the activation function
    x=tf.matmul(input,weights[-1])+biases[-1] #last layer is linear

    return x
'''

def multilayer_perceptron(x, weights, biases):
    if (len(weights)!=len(biases)): 
        return 'weights and biases must have the same size'

    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['w_out']) + biases['b_out']

    return out_layer


def main():

    #n_input, n_hidden_1, n_hidden_2, n_output = int(input('Enter n_input, n_hidden_1, n_hidden_2, n_output: '))
    n_input, n_hidden_1, n_hidden_2, n_output=1,5,2,1

    #weights
    weights={
        'w1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
        'w2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
        'w_out': tf.Variable(tf.random.normal([n_hidden_2, n_output])),
    }

    #biases
    biases={
        'b1': tf.Variable(tf.random.normal([n_hidden_1])),    
        'b2': tf.Variable(tf.random.normal([n_hidden_2])),
        'b_out': tf.Variable(tf.random.normal([n_output])),
    }

    '''general
    start = tf.constant([-1 for x in range( n_input )], dtype = tf.float32)
    end = tf.constant([1 for x in range( n_input )], dtype = tf.float32)
    x=tf.linspace(start, end, 10)'''
    x = np.linspace(-1, 1, 10, dtype=np.float32).reshape(-1, 1)
    y= multilayer_perceptron(x, weights, biases) #vettore di risultati, uno per ogni x che gli passo alla volta
    print(y)


if __name__=='__main__':
    main()
