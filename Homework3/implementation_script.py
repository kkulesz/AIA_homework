import numpy as np
import torch
import matplotlib.pyplot as plt

import utils

LEARNING_RATE = 1e-5
NUM_OF_STEPS = 1000
CONVERGENCE_THRESHOLD = 0.01


def task_1_function(X, sigma, num_of_steps, learning_rate, convergence_threshold):
    mu = torch.tensor((0., 0.), requires_grad=True)
    mu_history = [mu.detach().clone()]
    previous_loss = float("inf")  # set it to infinity, so we do not reach convergence at the very first step

    for i in range(num_of_steps):
        loss = utils.L(X, mu, sigma)
        loss.backward()
        mu.data = mu.data + learning_rate * mu.grad
        mu.grad.zero_()

        mu_history.append(mu.detach().clone())

        if abs(loss.data - previous_loss) <= convergence_threshold:
            print(f"Convergence reached at step={i}")
            break
        previous_loss = loss.data

    utils.vizualize(data, mu_history, sigma)


def task_2_function(X, sigma, num_of_steps, learning_rate, convergence_threshold):
    mu = torch.tensor((0., 0.), requires_grad=True)
    mu_history = [mu.detach().clone()]
    previous_loss = float("inf")  # set it to infinity, so we do not reach convergence at the very first step

    optimizer = torch.optim.Adam(params=[mu], lr=learning_rate)
    for i in range(num_of_steps):
        loss = -utils.L(X, mu, sigma)  # NOTE: we calculate "-L" because we want to perform gradient ASCENT instead of descent!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mu_history.append(mu.detach().clone())

        if abs(loss.data - previous_loss) <= convergence_threshold:
            print(f"Convergence reached at step={i}")
            break
        previous_loss = loss.data

    utils.vizualize(data, mu_history, sigma)


def task_3_function(X, sigma, num_of_steps, learning_rate, convergence_threshold, number_of_samples):
    mu = torch.tensor((0., 0.), requires_grad=True)
    mu_history = [mu.detach().clone()]
    # previous_loss = float("inf")  # set it to infinity, so we do not reach convergence at the very first step

    optimizer = torch.optim.Adam(params=[mu], lr=learning_rate)
    for i in range(num_of_steps):
        # draw which data points will be taken into batch
        indices = np.random.choice(len(X), size=number_of_samples, replace=False)
        samples = X[indices]

        loss = -utils.L(samples, mu, sigma)  # NOTE: we calculate "-L" because we want to perform gradient ASCENT instead of descent!
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mu_history.append(mu.detach().clone())

        # if abs(loss.data - previous_loss) <= convergence_threshold:
        #     print(f"Convergence reached at step={i}")
        #     break
        # previous_loss = loss.data

    utils.vizualize(data, mu_history, sigma)


if __name__ == "__main__":
    n = 10000  # number of samples (descrease the number if computations take too much time)
    mu, sigma = np.ones(2), 5 * np.eye(2)  # mean and standard deviation of ground truth distribution
    data = np.random.multivariate_normal(mu, sigma, n)  # sample n data points from the distribution

    # Cast those into torch's tensors so everything works well
    mu = torch.from_numpy(mu)
    sigma = torch.from_numpy(sigma)
    data = torch.from_numpy(data)

    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

    # task_1_function(data, sigma, NUM_OF_STEPS, LEARNING_RATE, CONVERGENCE_THRESHOLD)
    # task_2_function(data, sigma, NUM_OF_STEPS, LEARNING_RATE, CONVERGENCE_THRESHOLD)
    # subsets_sizes = [1, 5, 10, 100, 1000, 5000]
    # for num_of_samples in subsets_sizes:
    #     task_3_function(data, sigma, NUM_OF_STEPS, LEARNING_RATE, CONVERGENCE_THRESHOLD, num_of_samples)
