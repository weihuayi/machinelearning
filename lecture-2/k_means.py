from functools import reduce
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# noinspection PyTypeChecker
def k_means(data, kernel, step=0):
    """
    :param data: samples
    :param kernel: numeric or list
    :param step: int
    """
    # c: set groups, u: kernel vector
    if isinstance(kernel, list):
        c = [[sample] for sample in kernel]
        u = [data[sample] for sample in kernel]
    else:
        index = np.arange(len(data))
        np.random.shuffle(index)
        index = index[:kernel]
        c = [[sample] for sample in index]
        u = [data[sample] for sample in index]

    for i in range(step):
        # append all the vectors into different set groups.
        for sample_index in range(len(data)):
            dist_list = [np.linalg.norm(data[sample_index] - u[set_index]) for set_index in range(len(u))]
            set_index = np.argmin(np.array(dist_list))
            if sample_index not in c[set_index]:
                c[set_index].append(sample_index)

        # reset the kernel vector.
        for set_index in range(len(u)):
            sum_vector = reduce(lambda x, y: x + y, [data[sample_index] for sample_index in c[set_index]])
            u[set_index] = sum_vector / len(c[set_index])

    # matrix of samples after k-means.
    v = []
    for set_index in range(len(u)):
        v.append(data[c[set_index]])
    return c, u, v


# noinspection PyTypeChecker
def k_means_graph(data, kernel, step=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c, u, v = k_means(data=data, kernel=kernel, step=step)
    for index in range(len(c)):
        point, = ax.plot(v[index][:, 0], v[index][:, 1], 'o', markersize=10)
        ax.plot(u[index][0], u[index][1], '+', c=point.get_color(), markersize=10)


if __name__ == '__main__':
    watermelon = pd.read_csv('./watermelon.csv', sep='\t', header=0)

    k_means_graph(data=watermelon.values, kernel=3, step=5)

    pd.scatter_matrix(watermelon, diagonal='kde', alpha=.8, marker='o')
    plt.show()
