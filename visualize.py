import scipy.misc
import matplotlib.pyplot as plt

def plot_means(means):

    k = means.shape[0]

    rows = k // 5 + 1
    columns = min(k, 5)

    for i in range(k):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(scipy.misc.toimage(means[i].reshape(28, 28),
                                      cmin=0.0, cmax=1.0))
