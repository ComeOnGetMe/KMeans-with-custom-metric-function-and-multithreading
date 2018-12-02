import numpy as np
from PIL import Image
from numpy import linalg as LA
from sklearn.datasets import fetch_mldata
from kmeans import kmeans

mnist = fetch_mldata('MNIST original', data_home='./data')
train_dataset = mnist.data[:60000]
train_labels = mnist.target[:60000]
test_dataset = mnist.data[-10000:]
test_labels = mnist.target[-10000:]


def score(test_dataset, test_labels, prototypes, prototype_labels, pmetric):
    res = np.empty((len(test_dataset)), dtype=np.uint8)
    D = pmetric(test_dataset, prototypes)
    res = D.argmin(axis=1)
    s = np.mean(prototype_labels[res] == test_labels)
    return res, s


# TD
from mpdist import mpdist
from kmeans import kmeans
from tangentDistance import *

k = 100
prototypes = []
prototype_labels = []
m1TD = lambda X, Y: mpdist(X, Y, oneSideTD, ("tangentDistance", "ctypes", "numpy"))
for i in xrange(10):
    print 'clustering %d-th label' % (i)
    cur_train_dataset = mnist.data[mnist.target == i]
    cur_test_dataset = mnist.target[mnist.target == i]
    init_center = np.array(cur_train_dataset[:(k / 10)])
    c, cid, dist = kmeans(cur_train_dataset, init_center, metric=m1TD, pdist=True)
    for cc in c: prototypes.append(cc)
    prototype_labels += [i] * (k / 10)

cls, s = score(test_dataset, test_labels, prototypes, np.array(prototype_labels), m1TD)
print s
