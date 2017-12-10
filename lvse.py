import numpy
import pickle
from numpy import array
import matplotlib.pyplot as plt

f = open('./weights/err_c2.txt')
y = pickle.load(f)
f.close()
x=numpy.arange(0,1000)

y=numpy.reshape(y,(1000,4))
y = numpy.sum(y,axis=1)

y=array(y).ravel()
plt.plot(x,y)
plt.xticks(numpy.arange(min(x), max(x)+2, 100.0))
plt.title('Loss VS Epochs',fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss',fontsize=14)

plt.show()
