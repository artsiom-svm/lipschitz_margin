import model.SimpleANN as SimpleANN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def train(model, steps, batch_size, safety_factor= 0.005, model_path=None):    
    '''

    Train the {model} with steps and batch size. Model is trained on MNIST dataset<p>
    @param model{SimpleANN} model to train<br>
    @param steps{int} number of steps to train<br>
    @param batch_size{int} size of the batch per step<br>
    @param safety_factor{float} frequency to save the model, default 0.005<br>
    @param model_path{String} path to place updated mode, default None<br>
    @return tuple (accuracy, loss, steps) during the training<br>

    '''
    # read data set
    mnist_dataset = read_data_sets('MNIST_data', one_hot=True)
    # get frequency to store the model
    save = (int)(steps * safety_factor)
    # empty lists for statistics
    accuracy = []
    loss = []
    t = []
    # run saver for model
    saver = tf.train.Saver()
    begin = time.time()
    for step in range(0, steps):
        # get next training batch
        batch_x, batch_y = mnist_dataset.train.next_batch(batch_size)
        # train the model
        model.session.run(model.update_op, feed_dict={model.x_placehold: batch_x, model.y_placehold: batch_y})
        if step % save == 0:
            # get fixed test batch
            batch_x, batch_y = mnist_dataset.train.next_batch(512)
            # get loss and prediction
            l, y = model.session.run([model.loss_tensor, model.predict], feed_dict={model.x_placehold: batch_x, model.y_placehold: batch_y})
            # calculate the accuracy
            a = np.mean(np.array(y) == np.argmax(batch_y, axis=1))
            loss.append(l)
            accuracy.append(a)
            t.append(step)
            # save model if path provided
            if model_path:
                model.save(model_path)
    end = time.time()
    print("Time of training: %.3f", end - begin)
    # return collected statistics
    return (accuracy, loss, t)


if __name__ == '__main__':
    model_path = 'trained_models/margin=0/model.ckpt'
    model = SimpleANN.SimpleANN(784, 10, margin_rate=1.0, source=model_path)
    train(model, 65000, 64, model_path=model_path);