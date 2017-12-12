import numpy
import pickle
from numpy import array
import matplotlib.pyplot as plt

f = open('./weights/totalloss.txt')
y = pickle.load(f)
f.close()
x=numpy.arange(0,500)

y=numpy.reshape(y,(500,4))
y = numpy.sum(y,axis=1)

y=array(y).ravel()
plt.plot(x,y)
plt.xticks(numpy.arange(min(x), max(x)+2, 50.0))

plt.title('Loss VS Epochs',fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss',fontsize=14)

plt.show()
