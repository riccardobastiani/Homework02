import numpy as np


def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """
    ##############################
    ###     YOUR CODE HERE     ###
    ############################## 
    return 1 / (1 + np.exp(-x))   

def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: the predicted 

    Returns:
        softmax_scores: it's the matrix containing probability for each sample and each class. The shape is (N, K)
    """
    ##############################
    ###     YOUR CODE HERE     ###
    ##############################
    exp_x = np.exp(y - np.max(y))
    softmax_scores = exp_x / np.sum(exp_x)

    return softmax_scores

