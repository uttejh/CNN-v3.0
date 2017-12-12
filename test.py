import pickle
import numpy
# f = open('./weights/bhl.txt')
# bhl = pickle.load(f)
# f.close()

# print bhl

# f = open('./weights/b2.txt')
# b2 = pickle.load(f)
# f.close()

# print b2

# f = open('./weights/bFC.txt')
# bFC = pickle.load(f)
# f.close()

# print bFC

# f = open('./weights/b1.txt')
# b1 = pickle.load(f)
# f.close()

# print b1

f = open('./weights/totalloss.txt')
bhl = pickle.load(f)
f.close()
print numpy.max(bhl)