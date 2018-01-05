import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# noinspection PyTypeChecker
def agnes(data, kernel, label='avg'):
    """
    :param data: samples
    :param kernel: numeric or list
    :param label: string, 'min' or 'max' or 'avg'
    """

    # c: set groups
    c = [[sample_index] for sample_index in range(len(data))]

    # initialize distance matrix of set groups
    gram = data.dot(data.T)
    dist = np.diag(gram) + np.diag(gram).reshape(-1, 1) - 2 * gram
    dist = dist + np.max(dist) * np.eye(len(dist))
    dist_org = dist.copy()
    print(dist_org[[1, 2], [3, 4]])

    q = len(data)
    while q > kernel:
        i, j = divmod(np.argmin(dist), dist.shape[1])
        c[i] = c[i] + c[j]
        c.pop(j)

        dist = np.delete(dist, j, axis=0)
        dist = np.delete(dist, j, axis=1)

        for k in range(j + 1, dist.shape[1]):
            if label == 'min':
                dist[i, k] = np.min(dist_org[c[i], :][:, c[k]])
            elif label == 'max':
                dist[i, k] = np.max(dist_org[c[i], :][:, c[k]])
            else:
                dist[i, k] = np.mean(dist_org[c[i], :][:, c[k]])
        q -= 1

    # matrix of samples after k-means.
    v = []
    for set_index in range(kernel):
        v.append(data[c[set_index]])
    return c, v


# noinspection PyTypeChecker
def agnes_graph(data, kernel, label='avg'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c, v = agnes(data=data, kernel=kernel, label=label)
    for index in range(len(c)):
        ax.plot(v[index][:, 0], v[index][:, 1], 'o', markersize=10)


if __name__ == '__main__':
    watermelon = pd.read_csv('./watermelon.csv', sep='\t', header=0)

    agnes_graph(watermelon.values, kernel=3)

    pd.scatter_matrix(watermelon, diagonal='kde', alpha=.8, marker='o')
    plt.show()
