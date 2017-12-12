import numpy
import pickle
from numpy import array
import matplotlib.pyplot as plt

f = open('./weights1/totalloss.txt')
w1 = pickle.load(f)
f.close()

f = open('./weights2/totalloss.txt')
w2 = pickle.load(f)
f.close()

f = open('./weights3/totalloss.txt')
w3 = pickle.load(f)
f.close()

f = open('./weights4/totalloss.txt')
w4 = pickle.load(f)
f.close()

x=numpy.arange(0,4000)
y=[]
w1=numpy.reshape(w1,(1000,4))
w1 = numpy.sum(w1,axis=1)

y.append(w1)

w2=numpy.reshape(w2,(1000,4))
w2 = numpy.sum(w2,axis=1)

y.append(w2)

w3=numpy.reshape(w3,(500,4))
w3 = numpy.sum(w3,axis=1)

y.append(w3)

w4=numpy.reshape(w4,(1500,4))
w4 = numpy.sum(w4,axis=1)

y.append(w4)


y=array(y).ravel()
plt.plot(x,y)
plt.xticks(numpy.arange(min(x), max(x)+2, 400.0))
plt.title('Loss VS Epochs',fontsize=18)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss',fontsize=14)

plt.show()
