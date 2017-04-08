import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
  
    In other words, the network has the following architecture:
  
    input - fully connected layer - ReLU - fully connected layer - softmax
  
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4,
                 hidden_type='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
    
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
    
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {
            'W1': std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        if hidden_type in ['relu', 'sigm', 'tanh']:
            self.hidden_type = hidden_type
        else:
            raise ValueError('hidden_type should be "relu" or "sigm"'
                             'not {}'.format(hidden_type))

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
    
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
    
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
    
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        l2 = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################

        # forward prop
        layer1 = X.dot(W1) + b1  # NxD @ DxH + 1xH = NxH
        if self.hidden_type == 'relu':
            layer1_active = np.maximum(0, layer1)
        elif self.hidden_type == 'sigm':
            layer1_active = sigmoid(layer1)
        else:
            layer1_active = np.tanh(layer1)

        layer2 = layer1_active.dot(W2) + b2  # NxH @ HxC + 1xC = NxC
        l2 = layer2[:]  # they mean scores without softmaxing
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return l2

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################

        # forward prop
        stab_l2 = l2 - np.max(l2)  # for numeric stability
        exp_l2 = np.exp(stab_l2)
        sum_l2 = exp_l2.sum(axis=1)
        # uncomment if something get broken
        # loss = -np.log(exp_l2[np.arange(N), y] / sum_l2).sum()
        loss = (-stab_l2[np.arange(N), y] + np.log(sum_l2)).sum()
        loss /= N

        # reg
        loss += 0.5 * reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################

        # back prop
        dlog_l2 = 1
        dcorr_min_l2 = 1
        dsum_l2 = (1 / sum_l2) * dlog_l2
        dcorrect_l2 = -dcorr_min_l2
        dexp_l2 = exp_l2 * dsum_l2[:, np.newaxis]  # NxC

        # layer2 weights
        grads['W2'] = layer1_active.T.dot(dexp_l2)
        grads['b2'] = dexp_l2.sum(axis=0)
        dlayer1_active = dexp_l2.dot(W2.T)  # NxH

        # gradient updates on correct classes
        # it is pretty quick compared to .dot, but maybe there is
        # cool vectorized decision
        for i in range(N):
            grads['W2'][:, y[i]] += layer1_active[i] * dcorrect_l2
            grads['b2'][y[i]] += dcorrect_l2

            dlayer1_active[i, :] += W2[:, y[i]] * dcorrect_l2

        # layer1 weights
        if self.hidden_type == 'relu':
            dlayer1 = dlayer1_active[:]
            dlayer1[layer1 <= 0] = 0  # NxH
        elif self.hidden_type == 'sigm':
            dlayer1 = (1 - layer1_active) * layer1_active * dlayer1_active
        else:
            l1 = layer1 * 2
            # few tricks for numeric stability
            stab_const = np.max(l1)
            exp_l1_act_nom = np.exp(l1 - stab_const)
            exp_l1_act_denom = np.exp(l1 - stab_const/2)
            dlayer1 = (4 * exp_l1_act_nom
                       / (exp_l1_act_denom + np.exp(-stab_const/2))**2)\
                      * dlayer1_active

        grads['W1'] = X.T.dot(dlayer1)
        grads['b1'] = dlayer1.sum(axis=0)

        grads['W1'] /= N
        grads['b1'] /= N
        grads['W2'] /= N
        grads['b2'] /= N

        grads['W1'] += 0.5 * reg * 2 * W1
        grads['W2'] += 0.5 * reg * 2 * W2
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            rand_idxs = np.random.choice(num_train,
                                         np.minimum(batch_size, num_train),
                                         replace=False)
            X_batch = X[rand_idxs]
            y_batch = y[rand_idxs]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            for name in self.params:
                self.params[name] -= grads[name] * learning_rate
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
    
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
    
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        layer1 = X.dot(self.params['W1']) + self.params['b1']
        if self.hidden_type == 'relu':
            layer1_active = np.maximum(0, layer1)
        elif self.hidden_type =='sigm':
            layer1_active = sigmoid(layer1)
        else:
            layer1_active = np.tanh(layer1)
        layer2 = layer1_active.dot(self.params['W2']) + self.params['b2']
        layer2 -= np.max(layer2)  # for numeric stability of softmax
        layer2 = np.exp(layer2)
        layer2 /= layer2.sum(axis=1)[:, np.newaxis]
        y_pred = np.argmax(layer2, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
