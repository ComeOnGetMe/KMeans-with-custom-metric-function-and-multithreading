import tangentDistance as td
from sklearn.datasets import load_digits


data = load_digits().data
type(data[0]), data[0].shape, data[0].ndim
print td.oneSideTD(data[0], data[1], 8, 8)
