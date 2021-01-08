# nnfs custom 3


# forward und backward funktioniert bei dense, dropout, ReLU, softmax, crossentropy loss softmaxcrossentropyloss
# adam optimizer funktioniert
# implementing AdamW and regularization:
# keep regularization gradients and normal gradients separate and only combine them in the optimizer, so that
# AdamW and other Optimizers can distinguish between gradient from data loss and gradient from regularization penalty
# TODO: Adam and AdamW seems to work with reg_l2 ( no error), results are still to examined
# loss seems to not go down as wished, take a look at it
# SGD with momentum: TODO: add post_update_params or pre_update_params method ( one is missing)

# NOTE: tried to do regression with MSELoss and Softmax output or ReLU output
# shapes didnt match, used nnfs spiral data
# shapes were (300,) of y_true and (300,1) of y_pred,
# probably just because spiral_data isnt made for regression, but some kind of type,shape checking could be useful.
# maybe as a check_data method that can be used after finalize and before train
# or gets called in the beginning of train
# TODO: add accuracy classes for classification and for regression !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# compare training results from custom with training results from book code
# TODO: loss values seem to high / training to slow -> check optimizer and loss calculation + check if all optimizers work with l2 gradients
# TODO: warum ruiniert batch_size = 100 das training bei spiral_data? vllt. weil shuffling fehlt?
# -> problem tritt beim code aus dem buch auch auf


import numpy as np
import pickle
import os
import nnfs  # to have spiral data
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt  # for the report method
import cv2  # for image data

# input layer to store the input


class Layer_Input(object):
    def __init__(self):
        self.output = 0

    def forward(self, X):
        self.output = X  # TODO: how does this work?

# Eine dichte Lage von Neuronen


class Layer_Dense:

    # initialize weights and biases
    # TODO: None or 0 better for reg_l2 default?
    def __init__(self, n_inputs, n_neurons, reg_l2=None):
        # gives a n_neurons x n_inputs matrix
        self.weights = 0.01 * np.random.rand(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))  # gives a 1 x n_neurons matrix
        self.reg_l2 = reg_l2

    def forward(self, X):
        # save the inputs for the backward pass
        self.inputs = X
        # X is a matrix of samples x features(=inputs)
        self.output = np.dot(X, self.weights.T) + self.biases
        # dotof samples x inputs , n_neurons x n_inputs.T = dotof samples x inputs , n_inputs x n_neurons = samples x n_neurons

    def backward(self, dvalues):
        # method gets dvalues as a matrix with shape samles x neurons as the output is also shaped

        # derivative of dvalues toward biases is 1
        # and should have the same shape  as biases, which is:
        # 1 x neurons in this layer
        # but dvalues is samples x neurons:
        # -> sum dvalues column wise, do not collapse the array -> keepdims = True
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # derivative of dvalues toward weights is inputs
        # gradient should match shape of weights which is
        # neurons x inputs from previous layer
        # shape of dvalues is
        # samples x neurons
        # shape of inputs is
        # samples x inputs from previous layer
        # therefore: transpose dvalues to be neurons x samples
        # -> dotof: neurons x samples , samples x inputs from previous layer -> neurons x inputs from previous layer

        transposed_dweights = np.dot(self.inputs.T, dvalues)
        self.dweights = transposed_dweights.T  # transpose back again

        # calculate l2 penalty for weights and biases, if reg_l2
        if self.reg_l2 is not None:
            self.dweights_l2 = self.reg_l2 * 2 * self.weights
            self.dbiases_l2 = self.reg_l2 * 2 * self.biases

        # why do they keep the weights transposed? - just because they dont want them to be transposed during the forward pass

        # derivative of dvalues toward inputs is weights
        # should match the shape of the output of the previous layer
        # target shape: samples x weights
        # shape of dvalues is: samples x neurons
        # shape of weights is: neurons x weights
        # dotof samples x neurons , neurons x weights = samples x weights
        self.dinputs = np.dot(dvalues, self.weights)

# Dropout layer


class Layer_Dropout(object):

    def __init__(self, dropout_rate):
        # chance of a neuron getting deactivated i.e. "dropped out"
        self.dropout_rate = dropout_rate
    # performs a forward pass, randomly deactivating neurons
    # TODO: dropout should only be active for training. come up with a solution
    # probably wait unit building the model class

    def forward(self, inputs):
        # save the input values
        self.inputs = inputs
        # dropout sieve is randomly generated for every forward pass (= every batch)
        self.dropout_sieve = np.random.binomial(
            1, 1 - self.dropout_rate, inputs.shape)
        self.output = inputs * self.dropout_sieve

        # increase output according to the dropout rate, so that the mean of all outputs is the same with every dropout rate
        self.output = self.output / (1 - self.dropout_rate)
    # performs a backward pass, the dropped out neurons dont get a gradient of zero

    def backward(self, dvalues):
        # dropout(x) = x if dropout_sieve = 1
        # dropout(x) = 0 if dropout_sieve = 0

        # derivative if output with respect to input is therefore:
        # dinput(x) = 1 if dropout_sieve = 1
        # dinput(x) = 0 if dropout_sieve = 0

        # outputs have same shape as inputs -> dvalues have same shape as dinputs
        self.dinputs = dvalues * self.dropout_sieve


# simple Activation function
class Activation_ReLU(object):

    # perform a forward pass
    def forward(self, inputs):
        self.output = np.maximum(inputs, 0)
        # save inputs for the backward pass
        self.inputs = inputs

    # perform a backward pass
    def backward(self, dvalues):

        # derivative of dvalues towards inputs is:
        # 1 if dvalue > 0
        # 0 otherwise
        # we need to apply the chainrule here:
        self.dinputs = dvalues.copy()  # * 1
        self.dinputs[self.inputs < 0] = 0

# Softmax activation for classification (class confidences sum up to 1)


class Activation_Softmax(object):
    # softmax(x) = e^(xi) / sum( e^(i) for i in range(n))

    # perform a forward pass
    def forward(self, X):
        # X is a samples x neurons array
        # subtract to prevent
        power = np.exp(X - np.max(X, axis=1, keepdims=True))
        # exploding values
        # subtracting from every
        # input does not
        # change the outputs,
        # since they get normalised
        # by each other
        # weird but works
        # sum column-wise
        summed = np.sum(power, axis=-1, keepdims=True)
        # divide
        result = power / summed
        # calculate sum for each sample
        # calculate e to the power of value
        # divide
        # return result
        # TODO: seems to work correctly, not sure if there's a faster way
        self.output = result

    def backward(self, dvalues):

        # we want to compute the derivative of dvalues toward the inputs of the softmax function
        # we must compute the derivative of every dvalues with respect to every softmaxinput as every input influences every output
        # each output is a function of every single input
        # dSij with respect to input ik = Sij * ( crondelta(jk) - Sik)

        # each row of dvalues is a sample
        # -> we store the number of samples as this will be needed for gradient normalization
        samples = len(dvalues)  # counts the rows i.e. samples in dvalues
        # each column of dvalues is a neuron
        # as we get a jacobian matrix for every sample = row of dvalues, we need to compute these matrices row-wise
        # i.e. sample-wise
        # we create a list to store the results of every iteration
        jacobian_row_list = []
        # therefore we must iterate over the rows of dvalues -> for row in dvalues: ...
        for single_dvalue, single_output in zip(dvalues, self.output):

            # we get the second term in the jacobian matrix by forming the dot product of the row transposed and the row
            # for this we need to convert the row of single output to a column vector
            single_output_column = single_output.reshape(-1, 1)
            # .T in second argument converts column back to row
            sec_term = np.dot(single_output_column, single_output_column.T)
            # we get the first term in the jacobian matrix by taking the row and make it the diagonal of a matrix with the the shape: outputs x outputs
            # i.e. classes x classes
            first_term = np.diagflat(single_output)
            # then we can compute the difference between the first and the second term
            jacobian = first_term - sec_term
            # since we are interested in the influence on the loss function of the neurons in the layer before the softmax function,
            # we have to sum all partial derivatives towards that neuron output
            # but only in so far as we multiply it with its corresponding dvalues to follow the chain rule
            # this can be done by computing the dot product of the row vector in dvalues and the jacobian matrix, resulting in a row-vector
            # -> every sample yields a row-vector
            # jacobian_row = np.dot(single_dvalue, jacobian) # mine
            jacobian_row = np.dot(jacobian, single_dvalue)  # from the book
            # from the book: every input effects every output, so the derivative of the output wrt one input includes all paths over the other outputs
            # we derive not one output wrt to one input but every output wrt to one input(the sum of these derivatives is the dotproduct)
            # -> append these rowvectors to a list and in the end we form a new array with the shape samples x neurons
            jacobian_row_list.append(jacobian_row)
            # this then has the correct shape for the backward pass through the dense layer
        self.dinputs = np.array(jacobian_row_list)
        # now we have a 2D array with the shape samples x neurons containing the gradients
        # TODO: normalization seems to not be needed here, but why? maybe because we normalized in the Crossentropy loss and it would be a mistake
        # to divide by the number of samples again..

# Sigmoid activation "squishes" inputs into outputs between 0 and 1


class Activation_Sigmoid(object):
    pass

# Linear activation function for regression


class Activation_Linear(object):

    def forward(self, X):
        # remember inputs
        self.inputs = X
        self.output = X

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
# common loss class


class Loss(object):

    # calculate the loss
    def calculate(self, y_pred, y_true, regularization_loss=True):

        # sample losses are used to compute mean loss over all samples
        sample_losses = self.forward(y_pred, y_true)

        # data loss is only to have a metric for tracking training progress
        data_loss = np.mean(sample_losses)

        # if regularization_loss shall be calculated (default)
        if regularization_loss:
            # calculate total regularization loss
            regularization_loss = 0
            # loop over layers, saved in self.layers
            for layer in self.layers:
                regularization_loss += self.reg_loss_of_layer(layer)

            return data_loss, regularization_loss
        else:
            return data_loss

    # keeps track of accumulated loss over several batches(whatever is passed into it)
    def calculate_accumulated(self, y_pred, y_true):

        # always ignore reg loss
        data_loss = self.calculate(y_pred, y_true, regularization_loss=False)
        # mean * samples to have the sum of sums, instead of sum of means
        self.accumulated_loss += data_loss * len(y_pred)
        self.accumulated_samples += len(y_pred)

    # reset the accumulated sums of the loss object
    def accumulated_new_pass(self):
        self.accumulated_loss = 0
        self.accumulated_samples = 0

    # computes regularization loss of a layer
    def reg_loss_of_layer(self, layer):
        regularization_loss = 0
        # do not calculate reg_loss if layer has no weights i.e. is not trainable
        if not hasattr(layer, 'weights'):
            return 0

        # only compute if l2 != 0
        if layer.reg_l2 is not None:
            regularization_loss += layer.reg_l2 * \
                np.sum(layer.weights * layer.weights)
            regularization_loss += layer.reg_l2 * \
                np.sum(layer.biases * layer.biases)
            return regularization_loss
        else:
            return 0

    def remember_layers(self, layers):
        self.layers = layers


class Loss_BinaryCrossentropy(object):
    pass


class Loss_CategoricalCrossentropy(Loss):

    # calculate the loss from inputs - aka forward pass of the loss function
    def forward(self, y_pred, y_true):

        # clip y_pred to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # y_pred has shape samples x classes

        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred[range(len(y_pred)), y_true]

        # only if one hot labels
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=-1)

        # calculate negative logs
        neg_logs = -np.log(correct_confidences)
        return neg_logs

    # calculate the derivative of the loss function with respect to each of its inputs, being the softmax output
    def backward(self, dvalues, y_true):
        # dvalues are y_pred from the softmax function
        # count the samples for gradient normalization
        samples = len(dvalues)

        # check if labels are categorical
        # if so, convert them to one hot
        if len(y_true.shape) == 1:
            # count labels for np.eye method
            # use the first sample to count how many labels we have
            labels = len(dvalues[0])
            y_true = np.eye(labels)[y_true]

        # calculate gradient according to: dinput = - y_true / y_pred
        self.dinputs = - y_true / dvalues
        # normalize gradient, so that the sum of the gradient is invariant to the number of samples
        # here we divide
        # when backpropagating through the dense layers, we will compute the sum
        # this then is the same as computing the mean of the gradients
        self.dinputs = self.dinputs / samples

# MSE Loss


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        # mean of squared differences between y_pred and y_true
        sample_losses = (y_pred - y_true)**2
        return sample_losses
        # TODO: make this work with classification (shapes of one hot and sparse labels etc.)

    def backward(self, y_pred, y_true):
        # MSEloss(y_pred, y_true) = (y_pred - y_true)**2
        # dMSE / dy_pred = 2 * (y_pred - y_true) * 1
        self.dinputs = 2 * (y_pred - y_true)


class Activation_Softmax_Loss_CategoricalCrossentropy(object):

    # calculate gradients through Crossentropy and softmax at once, using the chain rule
    def backward(self, dvalues, y_true):

        # dvalues are the confidences aka softmax.output
        # shape is samples x classes just as the shape of dinuts shall be
        # derivative of dvalues with respect to the softmaxinput is y_pred - y_true

        # count how many samples there are for gradient normalization
        samples = len(dvalues)
        # if labels are sparse, turn them into one hot
        if len(y_true.shape) == 1:
            # count how many labels there are - use the first sample for it
            labels = len(dvalues[0])
            y_true = np.eye(labels)[y_true]
        self.dinputs = dvalues.copy()
        self.dinputs = dvalues - y_true
        # why do we normalize the gradient?
        # when later the dot product is performed in the backward of dense layers,
        # this is the same as suming the gradient for all samples together
        # which increases the gradient with more samples
        # making it necessacary to adjust the learning rate to batch-size
        # by dividing gradients here we avoid this, since dividing by # of samples, then sum these values
        # is to compute the mean of the values
        self.dinputs = self.dinputs / samples


# accuracy class for classification model
class Accuracy_Classification(object):

    # calculates the accuracy of predictions
    def calculate_accuracy(self, y_pred, y_true):
        pass
        # find out if y_true is one hot or sparse
        # if one hot, turn to sparse
        if len(y_true.shape) == 1:
            targets = y_true
        else:
            targets = np.argmax(y_true, axis=-1)

        predictions = np.argmax(y_pred, axis=-1)
        accuracy = np.mean(predictions == targets)
        return accuracy
    # accumulated accuracy calculation

    def calculate_accumulated(self, y_pred, y_true):

        # calculate for the batch, then multiply mean with number of samples and add it to the accumulated count
        batch_accuracy = self.calculate_accuracy(y_pred, y_true) * len(y_pred)
        # add number of samples to accumulated sample count
        self.accumulated_accuracy += batch_accuracy
        self.accumulated_samples += len(y_pred)

    # reset sums for accumulated accuracy
    def accumulated_new_pass(self):
        # reset the accumulated sums
        self.accumulated_accuracy = 0
        self.accumulated_samples = 0

# accuracy class for regression model


class Accuracy_Regression(object):

    # calculates accuracy of the predictions
    # default of precision is std(y_true/250)
    def calculate_accuracy(self, y_pred, y_true, precision=0.004):
        pass

        # tolerance is sd(y_true) / 250 = 0.004 by default
        tolerance = np.std(y_true) * precision
        abs_error = np.absolute(y_pred - y_true)
        accuracy = np.mean(abs_error <= tolerance)
        return accuracy

    def calculate_accumulated(self, y_pred, y_true, precision=0.004):

        # calculate accuracy of batch data
        batch_accuracy = self.calculate_accuracy(
            y_pred, y_true, precision) * len(y_pred)
        self.accumulated_accuracy += batch_accuracy
        self.accumulated_samples += len(y_pred)

    def accumulated_new_pass(self):
        self.accumulated_accuracy = 0
        self.accumulated_samples = 0

# Adam optimizer


class Optimizer_Adam(object):

    # initialize
    def __init__(self, *, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-7, decay=0, iterations=0):

        # set parameters as instance variables
        # learning rate is called alpha in many explanations
        self.learning_rate = learning_rate
        self.beta1 = beta1  # for first moment estimation
        self.beta2 = beta2  # for second moment estimation
        self.eps = 1e-7
        self.decay = decay
        self.iterations = iterations

    # perform before parameters are updated

    def pre_update_params(self):
        pass

    # update the parameters
    def update_params(self, layer):

        # if layer has no weights (= is not trainable), return
        if not hasattr(layer, 'weights'):
            return

        # trainable, so check if dweights_first_moment is initialized
        if not hasattr(layer, 'dweights_first_moment'):

            # initialize first_moment and second_moment for dweights and dbiases as zeros
            layer.dweights_first_moment = np.zeros_like(layer.dweights)
            layer.dweights_second_moment = np.zeros_like(layer.dweights)

            layer.dbiases_first_moment = np.zeros_like(layer.dbiases)
            layer.dbiases_second_moment = np.zeros_like(layer.dbiases)

        # if reg_l2 is not none, add the dreg_loss to the gradients of weights and biases here
        if layer.reg_l2 is not None:
            layer.dweights += layer.dweights_l2
            layer.dbiases += layer.dbiases_l2
        # estimate first moment (biased towards the past i.e. toward 0 in the beginning
        layer.dweights_first_moment = layer.dweights_first_moment * \
            self.beta1 + (1 - self.beta1) * layer.dweights
        layer.dbiases_first_moment = layer.dbiases_first_moment * \
            self.beta1 + (1 - self.beta1) * layer.dbiases

        # estimate second moment (biased towards 0)
        layer.dweights_second_moment = layer.dweights_second_moment * \
            self.beta2 + (1 - self.beta2) * layer.dweights**2
        layer.dbiases_second_moment = layer.dbiases_second_moment * \
            self.beta2 + (1 - self.beta2) * layer.dbiases**2

        # correct the bias of the moments towards the past i.e towards 0 in the beginning
        # -> result: moment estimation gets more biases towards the past as time passes ( = self.iterations increases)
        dweights_first_moment_corr = layer.dweights_first_moment / \
            (1 - self.beta1**(self.iterations+1)
             )  # self.iterations starts at 0, so add one
        # TODO: maybe init self.iterations as 0 and +1 in pre_update_params
        dbiases_first_moment_corr = layer.dbiases_first_moment / \
            (1 - self.beta1**(self.iterations+1))
        dweights_second_moment_corr = layer.dweights_second_moment / \
            (1 - self.beta2**(self.iterations+1))
        dbiases_second_moment_corr = layer.dbiases_second_moment / \
            (1 - self.beta2**(self.iterations+1))

        # update the parameters
        layer.weights += - (self.learning_rate / (1 + self.decay*self.iterations)) * dweights_first_moment_corr / \
            (np.sqrt(dweights_second_moment_corr) +
             self.eps)  # eps to prevent division by 0
        layer.biases += - (self.learning_rate / (1 + self.decay*self.iterations)) * \
            dbiases_first_moment_corr / \
            (np.sqrt(dbiases_second_moment_corr) + self.eps)

    # call once after every layer's parameters were updated
    def post_update_params(self):
        self.iterations += 1

# AdamW optimizer - modification of Adam, to generalize better, when regularization is used (from the blog post)


class Optimizer_AdamW(object):

    # initialize everything
    def __init__(self, *, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-7, decay=0, iterations=0):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # decay rate of moving average for first moment
        self.beta2 = beta2  # decay rate of moving average for second moment
        self.eps = eps
        self.decay = decay
        self.iterations = iterations

    def pre_update_params(self):
        pass

    def update_params(self, layer):

        # note: dweights and dbiases so far DO NOT include reg_l2 gradients, we include them in the final update of the parameters
        # check if the layer is trainable, if not, return
        if not hasattr(layer, 'weights'):
            return

        # if layer.reg_l2 attribute is None: set layer.dweights_l2 and layer.dbiases_l2 to 0
        if layer.reg_l2 is None:
            layer.dweights_l2 = 0
            layer.dbiases_l2 = 0

        # if layer has no first and second moment estimates, initialize them as zeros
        if not hasattr(layer, 'dweights_first_moment'):

            layer.dweights_first_moment = np.zeros_like(layer.dweights)
            layer.dbiases_first_moment = np.zeros_like(layer.dbiases)
            layer.dweights_second_moment = np.zeros_like(layer.dweights)
            layer.dbiases_second_moment = np.zeros_like(layer.dbiases)

        # now compute the estimates for the moments ( = the biased toward the past ones)
        layer.dweights_first_moment = layer.dweights_first_moment * \
            self.beta1 + (1 - self.beta1) * layer.dweights
        layer.dbiases_first_moment = layer.dbiases_first_moment * \
            self.beta1 + (1 - self.beta1) * layer.dbiases

        layer.dweights_second_moment = layer.dweights_second_moment * \
            self.beta2 + (1 - self.beta2) * layer.dweights**2
        layer.dbiases_second_moment = layer.dbiases_second_moment * \
            self.beta2 + (1 - self.beta2) * layer.dbiases**2

        # compute the estimates corrected from the bias toward the past (while the correction decays with more iterations)
        dweights_first_moment_corr = layer.dweights_first_moment / \
            (1 - self.beta1**(self.iterations+1)
             )  # self.iterations starts at 0, so we add 1
        dbiases_first_moment_corr = layer.dbiases_first_moment / \
            (1 - self.beta1**(self.iterations+1))

        dweights_second_moment_corr = layer.dweights_second_moment / \
            (1 - self.beta2**(self.iterations+1))
        dbiases_second_moment_corr = layer.dbiases_second_moment / \
            (1 - self.beta2**(self.iterations+1))

        # now we can update the parameters, finally including the gradient of the l2 regularization
        # normal adam update would be:
        #layer.weights += - (self.learning_rate / (1 + self.decay*self.iterations)) * (dweights_first_moment_corr / (np.sqrt(weights_second_moment_corr) + self.eps))

        # with the reg_l2 gradient added to the mix:
        layer.weights += - (self.learning_rate / (1 + self.decay*self.iterations)) * \
            (dweights_first_moment_corr /
             (np.sqrt(dweights_second_moment_corr) + self.eps) + layer.dweights_l2)

        layer.biases += - (self.learning_rate / (1 + self.decay*self.iterations)) * \
            (dbiases_first_moment_corr /
             (np.sqrt(dbiases_second_moment_corr) + self.eps) + layer.dbiases_l2)

    # once after all layers got their parameters updated
    def post_update_params(self):
        self.iterations += 1

# classic SDG optimizer with momentum


class Optimizer_SGDMomentum(object):

    # initialize optimizer
    def __init__(self, *, learning_rate=0.1, decay=1e-5, momentum=0, iterations=0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = iterations
        self.momentum = momentum

    # once before parameters are updated
    def pre_update_params(self):
        pass

    # updates the parameters
    def update_params(self, layer):

        # if layer has no weights, dont train it
        if not hasattr(layer, 'weights'):
            return

        # if layer has no momentum initialize it as zero
        if not hasattr(layer, 'dweights_momentum'):
            layer.dweights_momentum = np.zeros_like(layer.dweights)
            layer.dbiases_momentum = np.zeros_like(layer.dbiases)

        # update the momentums
        layer.dweights_momentum = layer.dweights_momentum * \
            self.momentum + (1 - self.momentum) * layer.dweights
        layer.dbiases_momentum = layer.dbiases_momentum * \
            self.momentum + (1 - self.momentum) * layer.dbiases

        if not hasattr(layer, 'dweights_l2'):
            # set dweights_l2 and dbiases_l2 to 0 to prevent error
            layer.dweights_l2 = 0
            layer.dbiases_l2 = 0
        # update the parameters of the layer
        layer.weights += - (self.learning_rate / (1 + self.decay*self.iterations)) * (
            layer.dweights_momentum + layer.dweights_l2)  # here we add reg gradients
        layer.biases += - (self.learning_rate / (1 + self.decay*self.iterations)) * (
            layer.dbiases_momentum + layer.dbiases_l2)  # no need to have momentum for this

    # once after parameters are updated
    def post_update_params(self):
        self.iterations += 1

# vanilla SGD optimizer


class Optimizer_SGDVanilla(object):

    # init
    def __init__(self, learning_rate=0.01, decay=5e-5):

        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        pass

    def update_params(self, layer):

        # if layer has no weights, dont train it
        if not hasattr(layer, 'weights'):
            return

        weight_gradients = layer.dweights
        bias_gradients = layer.dbiases
        if layer.reg_l2 is not None:
            weight_gradients += layer.dweights_l2
            bias_gradients += layer.dbiases_l2

        weight_updates = - weight_gradients * \
            (self.learning_rate / (1 + self.iterations*self.decay))
        bias_updates = - bias_gradients * \
            (self.learning_rate / (1 + self.iterations * self.decay))

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
# model class to combine the parts


class Model(object):

    # initialize a model
    def __init__(self):

        # list to store the layer
        self.layers = []
        # layer to store the inputs
        self.input_layer = Layer_Input()

    # set loss function, optimizer and accuracy objects for the model
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # adds a layer or list of layers to the models self.layers property
    def add(self, single_layer_or_list):
        try:
            for element in single_layer_or_list:  # list unpacking creates type confusion
                self.layers.append(element)
        except:
            self.layers.append(single_layer_or_list)

    # performs all steps necessary to setup the model after it has been specified
    def finalize(self):
        # setup the model, so that each layer has a prev and next attribute

        # -1 does not include last layer, 1 does not include first layer
        for i, layer in enumerate(self.layers[:-1]):
            layer.prev = self.layers[i-1]
            layer.next = self.layers[i+1]
        # correct properties for the first layer and the last layer
        self.layers[0].prev = self.input_layer
        self.layers[-1].next = self.loss
        self.layers[-1].prev = self.layers[-2]

        # check if activation is softmax and categorical crossentropy loss
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # we can modify the backward pass
            self.activation_loss = Activation_Softmax_Loss_CategoricalCrossentropy()

        # make the loss object remember the layers of the model to calculate regularization loss of all layers
        self.loss.remember_layers(self.layers)

    # trains the model on a dataset
    # TODO: what should be the default for validate every?
    def train(self, X, y, *, validation_data=None, epochs=1, batch_size=None, print_every=1, validate_every=1):

        # setup validation data for better readibility
        if validation_data is not None:
            X_val, y_val = validation_data

        # training loop
        for epoch in range(epochs):

            # announce new epoch
            print('-------------------------------------------------------------')
            print(f'Epoch: {epoch}')

            # calculate number of steps needed to forward the whole training data
            steps = 1  # default value
            if batch_size is not None:
                steps = len(X) // batch_size  # len(X) is number of samples
                if steps * batch_size < len(X):
                    steps += 1

            for step in range(steps):

                # slice the batch or pass the whole training dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # slice a batch
                else:
                    batch_X = X[batch_size * step: batch_size * (step + 1)]
                    batch_y = y[batch_size * step: batch_size * (step + 1)]

                # set the batch data as self.input_layer.output
                self.input_layer.output = batch_X

                # forward pass
                self.forward()
                # calculate data_loss and regularization_loss if needed for the batch

                # print if needed
                if not (step % print_every) or step == steps:  # step+1 to start counting at zero
                    # calculate loss
                    data_loss, regularization_loss = self.loss.calculate(
                        self.layers[-1].output, batch_y)
                    total_loss = data_loss + regularization_loss
                    # calculate accuracy
                    # print loss and accuracy
                    accuracy = self.accuracy.calculate_accuracy(
                        self.layers[-1].output, batch_y)
                    print(
                        f'E: {epoch}',
                        f'step: {step}',
                        f'loss: {total_loss:.3f}',
                        f'data loss: {data_loss:.3f}',
                        f'reg loss: {regularization_loss:.3f}',
                        f'acc: {accuracy:.3f}'
                    )

                # backward pass
                # if we have a softmax activation function and Categorical Crossentropy Loss, we can take the shortcut
                self.backward(self.layers[-1].output, batch_y)

                # update parameters
                # pre update params
                self.optimizer.pre_update_params()
                for layer in self.layers:
                    self.optimizer.update_params(layer)
                # post update parameters
                self.optimizer.post_update_params()

                # validate if needed
                # validate at the end of every epoch
                if not (step % validate_every) or step == steps:

                    # perform a forward pass with the validation data
                    # with batches if needed
                    val_steps = 1
                    if batch_size is not None:
                        val_steps = len(X_val) // batch_size
                        if val_steps * batch_size < len(X_val):
                            val_steps += 1

                    # reset accumulated loss counter
                    self.loss.accumulated_new_pass()
                    # reset the accumulated accuracy counter
                    self.accuracy.accumulated_new_pass()

                    for val_step in range(val_steps):

                        # slice the batches
                        if batch_size is None:
                            batch_X_val = X_val
                            batch_y_val = y_val
                        else:
                            batch_X_val = X_val[batch_size *
                                                val_step: batch_size * (val_step + 1)]
                            batch_y_val = y_val[batch_size *
                                                val_step: batch_size * (val_step + 1)]

                        # set the input layer output to the batch data
                        self.input_layer.output = batch_X_val
                        # forward pass
                        self.forward()
                        # calculate loss and accuracy
                        self.loss.calculate_accumulated(
                            self.layers[-1].output, batch_y_val)
                        self.accuracy.calculate_accumulated(
                            self.layers[-1].output, batch_y_val)

                    # after all batches of val data are passed forward,
                    # divide accumulated loss by accumulated samples to have mean loss for val data
                    mean_val_loss = self.loss.accumulated_loss / self.loss.accumulated_samples
                    # calculate mean accuracy over entire val data
                    mean_val_accuracy = self.accuracy.accumulated_accuracy / \
                        self.accuracy.accumulated_samples

                    print(
                        f'Epoch: {epoch}, step: {step}',
                        f'validation loss: {mean_val_loss:.3f}',
                        f'validation accuracy: {mean_val_accuracy:.3f}',
                        '[VALIDATION DATA]'

                    )

    # performs a forward pass through the whole network

    def forward(self, X=None):  # TODO: seems to work
        if X is not None:  # allows for a manual forward pass outside of the train method
            self.input_layer.output = X  # TODO: might slow down unnecessarily
        for layer in self.layers:
            layer.forward(layer.prev.output)

    # performs a backward pass through the whole network
    def backward(self, y_pred, y_true):
        # backward through the loss
        # if loss is categorical crossentropy and last layer is softmax:
        if isinstance(self.loss, Loss_CategoricalCrossentropy) and \
                isinstance(self.layers[-1], Activation_Softmax):

            self.activation_loss.backward(y_pred, y_true)
            # set the dinputs of softmaxlayer to dinputs of combined-object
            # and start backwardpass at the layer before the softmax function
            self.layers[-1].dinputs = self.activation_loss.dinputs
            # exclude last layer(=Softmax) in backwards loop
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

        # backward pass without softmax and categorical cross entropy
        else:
            # compute self.loss.dinputs
            self.loss.backward(self.layers[-1].output, y)
            for layer in reversed(self.layers):  # include all layers
                layer.backward(layer.next.dinputs)


# ---------END OF CLASSES----------------------------------------------
