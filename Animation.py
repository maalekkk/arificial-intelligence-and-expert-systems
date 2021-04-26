import matplotlib.pyplot as plt


def distribution_plot(distribution, distribution_org, filename):
    plt.clf()
    plt.plot(range(len(distribution)), distribution, "r")
    plt.plot(range(len(distribution_org)), distribution_org, "b")
    plt.savefig('./plots/' + filename)


def track_plot(coords_test, reference_test, output, filename):
    plt.clf()
    plt.plot(coords_test[:, 0], coords_test[:, 1], "r+")
    plt.plot(reference_test[:, 0], reference_test[:, 1], "bo")
    plt.plot(output[:, 0], output[:, 1], "go")
    plt.savefig('./plots/' + filename)
