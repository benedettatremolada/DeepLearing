import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(0)


def multilayer_perceptron(x, weights, biases):
    if (len(weights)!=len(biases)): 
        return 'weights and biases must have the same size'

    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['w2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['w_out']) + biases['b_out']

    return out_layer


def main():

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

    x = np.linspace(-1, 1, 10, dtype=np.float32).reshape(-1, 1)
    y1= multilayer_perceptron(x, weights, biases)
    print(y1)


    # Sequential model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_hidden_1, activation='sigmoid', input_dim=1))
    model.add(tf.keras.layers.Dense(n_hidden_2, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(n_output, activation='linear'))
    model.summary()

    '''model = keras.Sequential(
        [
            keras.layers.Dense(n_hidden_1, activation="sigmoid",input_dim=n_input),
            keras.layers.Dense(n_hidden_2, activation="sigmoid"),
            keras.layers.Dense(n_output, activation="linear", ),
        ]
        )
    model.set_weights([  weights[0],biases[0],
                    weights[1],biases[1],
                    weights[2],biases[2],
                    ])'''
    

    # assign parameters from previous model
    model.set_weights([weights["w1"], biases["b1"],
                    weights["w2"], biases["b2"],
                    weights["w_out"], biases["b_out"]])
    y2 = model.predict(x) 

    if not np.allclose(y1, y2):
        raise ValueError("results do not match")
    else:
        raise "results match"
    



if __name__=='__main__':
    main()
