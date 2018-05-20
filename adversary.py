import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# @author Arsion Balakir
# This file is designed to perform testing of pre-trained modesl on adversary attack with gaussian noise


def test(model, dataset, batch_size, noise):
    '''

    Test {model} on performance in {dataset} with gaussian noise<p>
    @param model {SimpleANN} model that is tested on<br>
    @param dataset {Dataset} set to test on, MNIST from tf library<br>
    @param batch_size {int} size of the batch to test on <br>
    @param noise {float} value of the gaussian noise, [0,1]<br>
    @return {dict} two keys 'accuracy' and 'loss' from the test

    '''
    x, y = dataset.train.next_batch(batch_size)
    # add noise
    gaussian_noise = np.random.normal(size=x.shape) * noise
    x = x + gaussian_noise
    x = np.clip(x, 0.0, 1.0)
    # run on model
    loss, y_hat = model.session.run([model.loss_tensor, model.predict], feed_dict={model.x_placehold: x, model.y_placehold: y})
    accuracy = np.mean(np.array(y_hat) == np.argmax(y, axis=1))
    # return dict of the result
    return {"accuracy": accuracy, "loss": loss}


def pair_test(model_margin, model_marginless, dataset, batch_size):
    '''

    Performce paired test on adversary attack in form of gaussian noise
    on both models, plot the graphs of their accuracies and return tuple <p>
    @param model_margin {SimpleANN} model trained with margin<br>
    @param model_marginless {SimpleANN} model trained witout margin loss<br>
    @param dataset {Dataset} set to test on, used MNIST tf set<br>
    @param batch_size {int} size of the batch to test on<br>
    @return tuple (accuracy_margin{list}, accuracy_marginless{list})

    '''
    # linear spreaded noise from 0 to 1
    noises = np.arange(0.0, 1, 0.01, dtype=np.float64)
    
    acc_margin = []
    acc_less = []
    for noise in noises:
        # run test on each model, store the accuracies
        d = test(model_margin, dataset, batch_size, noise)
        acc_margin.append(d['accuracy'])
        d = test(model_marginless, dataset, batch_size, noise)
        acc_less.append(d['accuracy'])
    
    # plot labels for graph
    plt.xlabel('gaussian noise')
    plt.ylabel('Accuracy, %')
    # plot both curves
    plt.plot(np.array(acc_less) * 100, noises, label='w/out lipschitz margin')
    plt.plot(np.array(acc_margin) * 100, noises, label='w/ lipschitz margin')
    # set legend, show the graph
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    # return tuple of accuracies
    return (acc_margin, acc_less)