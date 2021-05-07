import numpy as np


def calc_error(ref_data, output):
    assert (len(ref_data) == len(output))
    return np.sum((output - ref_data) ** 2, axis=1) ** 0.5


# def calc_distribution(ref_data, output):
#     raw_count, raw_bins_count = np.histogram(ref_data, bins=30)
#     raw_pdf = raw_count / sum(raw_count)
#     raw_cdf = np.cumsum(raw_pdf)
#
#     net_count, net_bins_count = np.histogram(output, bins=30)
#     net_pdf = net_count / sum(net_count)
#     net_cdf = np.cumsum(net_pdf)
#
#     return raw_cdf, net_cdf
