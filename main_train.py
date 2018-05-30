import model.SimpleANN as SimpleANN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


def train(model, steps, batch_size, safety_factor= 0.02, model_path=None, offset=0):    
    '''

    Train the {model} with steps and batch size. Model is trained on MNIST dataset<p>
    @param model{SimpleANN} model to train<br>
    @param steps{int} number of steps to train<br>
    @param batch_size{int} size of the batch per step<br>
    @param safety_factor{float} frequency to save the model, default 0.005<br>
    @param model_path{String} path to place updated mode, default None<br>
    @param offset{int} offset from step nuber for history<br>
    @return tuple (accuracy, loss, steps) during the training<br>

    '''
    # read data set
    mnist_dataset = read_data_sets("MNIST_data/", one_hot=True)
    # get frequency to store the model
    save = (int)(steps * safety_factor)
    # empty lists for statistics
    accuracy = []
    loss = []
    t = []
    # run saver for model
    saver = tf.train.Saver()
    begin = time.time()
    for step in range(offset, steps + offset):
        # get next training batch
        batch_x, batch_y = mnist_dataset.train.next_batch(batch_size)
        # train the model
        model.session.run(model.update_op, feed_dict={model.x_placehold: batch_x, model.y_placehold: batch_y})
        if (step + 1) % save == 0:
            # get fixed test batch
            batch_x, batch_y = mnist_dataset.test.next_batch(512)
            # get loss and prediction
            l, y = model.session.run([model.loss_tensor, model.predict], feed_dict={model.x_placehold: batch_x, model.y_placehold: batch_y})
            # calculate the accuracy
            a = np.mean(np.array(y) == np.argmax(batch_y, axis=1))
            loss.append(l)
            accuracy.append(a)
            t.append(step)
            print("t = %d, accuracy = %f, loss = %f" % (step + 1, a, l))
            # save model if path provided
            if model_path:
                model.save(model_path)
        
    end = time.time()
    print("Time of training: %.3f" % (end - begin))
    # return collected statistics
    return (accuracy, loss, t)


def train_marginless(source_path, steps, offset=0):
    '''

    Trains model with margin = 0.0<br>
    @return tuple from train function<br>
    
    '''
    with tf.Graph().as_default():
        model_path = 'trained_models/margin=0/model.ckpt'
        model = SimpleANN.SimpleANN(784, 10, margin_rate=0.0, source=source_path)
        result = train(model, steps, 64, model_path=model_path, offset=offset);        
    return result

def train_margin(source_path, steps, offset=0):
    '''

    Trains model with margin = 0.0<br>
    @return tuple from train function<br>

    '''
    with tf.Graph().as_default():
        model_path = 'trained_models/margin=1/model.ckpt'
        model = SimpleANN.SimpleANN(784, 10, margin_rate=1.0, source=source_path)
        result = train(model, steps, 64, model_path=model_path, offset=offset);        
    return result

def train_models():
    steps = 10000
    d2 = train_margin(None, steps)    
    d3 = train_marginless(None, steps)
    return (d2, d3)

if __name__ == '__main__':
    train_models()