import matplotlib.pyplot as plt


def distribution_plot(distribution, distribution_org, filename):
    plt.clf()
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.plot(range(len(distribution)), distribution, "r", label='network output')
    plt.plot(range(len(distribution_org)), distribution_org, "b", label='reference data')
    plt.savefig('./plots/' + filename)


def track_plot(coords_test, reference_test, output, filename):
    plt.clf()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(coords_test[:, 0], coords_test[:, 1], "r+", markersize=1.5, label='raw data')
    plt.plot(reference_test[:, 0], reference_test[:, 1], "bo", markersize=1, label='network output')
    plt.plot(output[:, 0], output[:, 1], "go", markersize=1.5, label='reference data')
    plt.savefig('./plots/' + filename)
