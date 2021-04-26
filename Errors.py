def calc_error(data, output):
    assert (len(data) == len(output))
    error = []
    for i in range(len(data)):
        testing_xy = data[i]
        output_xy = output[i]
        error.append((((testing_xy[0] - output_xy[0]) ** 2) +
                      ((testing_xy[1] - output_xy[1]) ** 2)) ** 0.5)
    return error


def calc_distribution(data, output):
    errors = calc_error(data, output)
    distribution = []
    error = 0
    # TODO
    while len(distribution) == 0 or distribution[-1] < len(errors):
        distribution.append(len(list(filter(lambda e: e < error, errors))))
        error += 1
    distribution = [x / len(errors) for x in distribution]
    return distribution
