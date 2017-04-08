import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
  
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)  # at one image
        scores -= np.max(scores)  # for numerical stability
        denom = np.exp(scores).sum()
        num = np.exp(scores[y[i]])
        loss += -np.log(num/denom)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += X[i] * (num/denom - 1)
            else:
                dW[:, j] += X[i] * np.exp(scores[j])/denom

    loss /= num_train
    dW /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW += 0.5 * reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    # loss = 0.0
    # dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores)  # for numeric stability
    scores = np.exp(scores)

    loss_denom = scores.sum(axis=1)
    loss = -np.log(scores[np.arange(y.size), y] / loss_denom).sum()

    ratios = scores / loss_denom[:, np.newaxis]
    ratios[np.arange(y.size), y] -= 1
    dW = X.T.dot(ratios)

    dW /= num_train
    loss /= num_train

    loss += 0.5 * reg * np.sum(np.square(W))
    dW += 0.5 * reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
