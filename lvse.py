import numpy
import pickle
from numpy import array
import matplotlib.pyplot as plt

f = open('./weights/totalloss.txt')
y = pickle.load(f)
f.close()
x=numpy.arange(0,500)
font = {'family' : 'ubuntu medium',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
y=numpy.reshape(y,(500,4))
y = numpy.sum(y,axis=1)

y=array(y).ravel()
lines=plt.plot(x,y)
plt.xticks(numpy.arange(min(x), max(x)+2, 100.0))

plt.title('Loss VS Epochs',fontsize=30)
plt.xlabel('Epoch', fontsize=26)
plt.ylabel('Loss',fontsize=26)
plt.setp(lines, color='b', linewidth=2.0)
plt.show()
