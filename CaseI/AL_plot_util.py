import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_approximation2(gpr, X_mesh_plot, X_data_scaled, Y_data_scaled, feature_name, X_next=None, show_legend=False):
    mu, std = gpr.predict(X_mesh_plot, return_std=True)
    count = 0
    for jj in range(len(X_mesh_plot[1, :])):
        for kk in range(jj + 1, len(X_mesh_plot[1, :])):
            count = count + 1
            fig = plt.figure(figsize=(8, 8))
            # ax = plt.axes(projection='3d')
            ax = plt.subplot(2, 3, count, projection='3d')

            print(np.size(X_data_scaled[:, jj]))
            print(np.size(X_data_scaled[:, jj]))

            ax.plot3D(X_mesh_plot[:, jj], X_mesh_plot[:, kk], mu.flatten(), 'bo', label='Surrogate function')
            ax.plot3D(X_data_scaled[1:100, jj], X_data_scaled[1:100, kk], Y_data_scaled[1:100].squeeze(), 'kx',
                      label='Noisy samples')
            ax.set_xlabel(feature_name[jj])
            ax.set_ylabel(feature_name[kk])

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()


def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x) + 1)

    x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Distance between consecutive x\'s')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, 'ro-')
    plt.xlabel('Iteration')
    plt.ylabel('Best Y')
    plt.title('Value of best selected sample')