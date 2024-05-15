import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

tf.random.set_seed(0)

# Truth data
def f(x):
    return x * 3.0 + 2.0

#Artificial data generation
def generate_data():
    x = tf.linspace(-2, 2, 200)
    x = tf.cast(x, tf.float32)

    noise = tf.random.normal(shape=[200], mean=0.0, stddev=1.0)
    y=f(x)
    y_noise = f(x) + noise
    return x, y, y_noise

# My custom model
class MyModel(tf.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b


# My custom model with Keras
class MyKerasModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x, training=False):
        return self.w * x + self.b


# MSE loss function
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# TRAIN FUNCTION Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, learning_rate):

    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


# Define a training loop
def report(model, loss):
    return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss={loss:2.5f}"


def training_loop(model, x, y, epochs):

    weights = [] # Collect the history of W-values and b-values to plot later
    biases = []

    for epoch in epochs:
        # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.1)

        # Track this before I update
        weights.append(model.w.numpy())
        biases.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print(f"Epoch {epoch:2d}:")
        print("    ", report(model, current_loss))

    return weights, biases

def main():


    '''DATA GENERATION'''

    x, y, y_noise=generate_data()

    plt.figure(1)
    plt.plot(x, y, "orange", label="Ground Truth Model Data")
    plt.scatter(x, y_noise, label="Data (with noise)")
    plt.title('Data generation')
    plt.xlabel('x')
    plt.ylabel('f(x) = 3 * x + 2')
    plt.legend()
 

    '''LINEAR FIT'''

    # Custom model allocation (no trained)
    model = MyModel()
    ypred = model(x)

    plt.figure(2)
    plt.plot(x, y, "orange", label="Ground Truth Model Data")
    plt.scatter(x, y_noise, label="Data (with noise)")
    plt.plot(x, ypred, "green", label="Untrained predictions")
    plt.title("Untrained Model")
    plt.xlabel('x')
    plt.legend()
    plt.show()

    '''TRAINING LOOP'''

    # Training loop
    current_loss = loss(y, model(x))
    print("Untrained model loss: %1.6f" % loss(model(x), y).numpy())
    print(f"Starting:")
    epochs = range(10) 
    print("    ", report(model, current_loss))
    weights, biases = training_loop(model, x, y, epochs)
    print("Trained loss: %1.6f" % loss(model(x), y).numpy())

    plt.figure(3)
    plt.plot(epochs, weights, 'r', label='weights')
    plt.plot(epochs, [3.0] * len(epochs), 'r--', label = "True weight")
    plt.plot(epochs, biases, 'b', label='bias')
    plt.plot(epochs, [2.0] * len(epochs), "b--", label="True bias")
    plt.legend()


    '''POST-FIT'''

    plt.figure(4)
    plt.plot(x, y, "orange", label="Ground Truth Model Data")
    plt.scatter(x, y_noise, label="Data (with noise)")
    plt.plot(x, ypred, "green", label="Untrained predictions")
    plt.plot(x, model(x), "red", label="Trained predictions")
    plt.title("Functional API")
    plt.xlabel('x')
    plt.legend()
    plt.show()


    '''KERAS MODEL API'''

    keras_model = MyKerasModel()
    #weights, biases = training_loop(keras_model, x, y, epochs)
    keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                        loss=tf.keras.losses.mean_squared_error)
    keras_model.fit(x, y, epochs=10, batch_size=len(x))

    plt.figure()
    plt.plot(x, y, "orange", label="Ground Truth Model Data")
    plt.scatter(x, y_noise, label="Data (with noise)")
    plt.plot(x, ypred, "green", label="Untrained predictions")
    plt.plot(x, keras_model(x), "red", label="Trained predictions")
    plt.title("Keras Model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
