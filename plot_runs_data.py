import pickle
import sys

import matplotlib.pyplot as plt

quartiles = pickle.load(open(sys.argv[1], 'rb'))
print(quartiles)
# Draw a plot
for quartile in quartiles:
    (median, pos, neg) = zip(*quartile)
    print(median)
    plt.errorbar(x=range(0, 1000000, 50000),
                 y=median, yerr=(pos, neg))
plt.show()
