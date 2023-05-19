import numpy as np
import torch
import matplotlib.pyplot as plt

plot_range = [-1, 2]  # -5, 5 originally


def L(X, mu, sigma):
    """
    Computes the log-likelihood over a dataset X for an estimated normal distribution parametrized
    by mean mu and covariance sigma

    X : Tensor
        A data matrix of size n x 2
    mu: Tensor of size 2
        a tensor with two entries describing the mean
    sigma: Tensor of size 2x2
        covariance matrix
    """
    diff = X - mu
    z = -0.5 * diff @ sigma.inverse() * diff
    return z.sum()


def vizualize(X, mus, sigma):
    """
    Plots a heatmap of a likelihood evaluated for different mu.
    It also plots a list of gradient updates.

    X : Tensor
        A data matrix of size n x 2
    mus: list[Tensor]
        A list of 2D tensors. The tensors should be detached from and on CPU.
    sigma: Tensor of size 2x2
        covariance matrix
    """
    loss = lambda x, y: L(X, torch.tensor([x, y]), sigma)
    loss = np.vectorize(loss)
    space = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(space, space)
    zs = np.array(loss(np.ravel(x), np.ravel(y)))
    z = zs.reshape(x.shape)
    plt.pcolormesh(x, y, z)

    mu_x, mu_y = zip(*mus)
    plt.plot(mu_x, mu_y)
    plt.xlim(plot_range)
    plt.ylim(plot_range)
    plt.show()
