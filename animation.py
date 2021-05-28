import matplotlib.pyplot as plt
import numpy as np


def calc_and_plot_distribution(distribution_org, distribution, filename):
    plt.clf()
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.gca().set_xlim(0, 2000)
    plt.gca().set_ylim(0, 1)
    plt.title(filename)

    raw_count, raw_bins_count = np.histogram(distribution_org, bins=30)
    raw_pdf = raw_count / sum(raw_count)
    raw_cdf = np.cumsum(raw_pdf)

    plt.plot(raw_bins_count[1:], raw_cdf, label='raw data', c='blue')

    net_count, net_bins_count = np.histogram(distribution, bins=30)
    net_pdf = net_count / sum(net_count)
    net_cdf = np.cumsum(net_pdf)

    plt.plot(net_bins_count[1:], net_cdf, label='network output', c='red')

    plt.legend()
    plt.savefig('./plots/cdf/cdf_' + filename)
    return net_cdf


def plot_track(coords_test, reference_test, output, filename):
    plt.clf()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(coords_test[:, 0], coords_test[:, 1], 'r+', markersize=1.5, label='raw data')
    plt.plot(reference_test[:, 0], reference_test[:, 1], 'bo', markersize=1, label='reference data')
    plt.plot(output[:, 0], output[:, 1], 'go', markersize=1.5, label='network output')
    plt.title(filename)
    plt.legend()
    plt.savefig('./plots/track/track_' + filename)
