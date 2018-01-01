import numpy  	# Python's mathematical library
import Image	# To convert image to pixels
from procedures import *
import time		# To calculate time
import pickle	# To write and read files
import matplotlib.pyplot as plt 	# Graphical representation
numpy.set_printoptions(threshold=numpy.nan)


numOfFiltersLayer1 = 20
numOfFiltersLayer2 = 40

# height and width of the filters
fsize = 5

alpha = 0.1
count = 0.

numofInputImages = 4

numOfInputs1 = 28*28		
numOfOutputs1 = 24*24

numOfInputs2 = numOfFiltersLayer1*12*12		
numOfOutputs2 = 8*8		

numOfHiddenNeurons1 = 200
numOfHiddenNeurons2 = 200

numOfOutputNeurons = 2

target=[]
# epochs = 1000
for i in range(100):
	target.append(0.)
for i in range(100):
	target.append(1.)
for i in range(100):
	target.append(2.)
for i in range(100):
	target.append(3.)

# Activation Function - Sigmoid
def sigmoid(x):
	out = 1/(1+numpy.exp(-x))
	return out

# Sigmoid Derivative
def derivative(x):
	return x*(1-x)

# Z-score normalization (also called batch-normalization?)
def zscore(x):
	normalized = []
	length = x.shape[1]
	
	# (x-mean)/standard deviation
	for i in range(length):
		col = x[i]

		mean = numpy.mean(col)
		std = numpy.std(col)

		if std !=0:
			newval = (col - mean)/(std)
			normalized.append(newval)
		else:
			normalized.append(col)

	return normalized


# Reads and converts the input images into array of pixels
def readAllImages():
	data = []
	for i in range(numofInputImages):
		for j in range(100):

			name = './testDataset/'+str(i)+' ('+ str(j+1) +').png'
			image = Image.open(name)

			im = numpy.array( image, dtype="double" ) 

			# Normalizing data so that each column of Z has mean 0 and standard 1
			# also called as Z-score normalization
			im = zscore(im)

			data.append(im)

	return data

imagedata = readAllImages() # array of pixel data of all the input data 

p = Procedures()

filters1 = []
filters2 = []

err_hl = []
err_FC = []
err_c2 = []


f = open('./weights/filters1.txt')
filters1 = pickle.load(f)
f.close()

f = open('./weights/filters2.txt')
filters2 = pickle.load(f)
f.close()

f = open('./weights/FC_to_HL.txt')
weights_FC = pickle.load(f)
f.close()

f = open('./weights/HL_to_output1.txt')
weights_HL1 = pickle.load(f)
f.close()

f = open('./weights/HL_to_output2.txt')
weights_HL2 = pickle.load(f)
f.close()

f = open('./weights/b1.txt')
b1 = pickle.load(f)
f.close()

f = open('./weights/b2.txt')
b2 = pickle.load(f)
f.close()

f = open('./weights/bhl1.txt')
bhl1 = pickle.load(f)
f.close()

f = open('./weights/bhl2.txt')
bhl2 = pickle.load(f)
f.close()

f = open('./weights/bFC.txt')
bFC = pickle.load(f)
f.close()

totalloss = []


# Start the timer
start = time.time()

epochs = 500


for iterat_image in range(400):

	# Read one input at a time
	input_data = imagedata[iterat_image]


	# -----------------------------------------------------------------------------------------------
	#                     CONVOLUTION --> SIGMOID (Activation Fn) --> POOLING (FIRST ITERATION)
	# -----------------------------------------------------------------------------------------------

	# -------------------------------------- CONVOLUTION --------------------------------------------

	input_data_3d = numpy.reshape(input_data, (1,28,28))
	# print input_data_3d
	convolution_layer_1 = p.convolution(input_data_3d, filters1, numOfFiltersLayer1, 1, fsize, b1)		

	convolution_layer_1_shape = array(convolution_layer_1).shape

	# Batch normalization
	convolution_layer_1_values=[]
	for i in range(convolution_layer_1_shape[0]):
		convolution_layer_1_values.append(zscore(convolution_layer_1[0]))

	# -------------------------------------- SIGMOID ACTIVATION --------------------------------------------

	sigmoid_convLayer_1 = sigmoid(array(convolution_layer_1_values))

	# -------------------------------------- POOLING --------------------------------------------

	pool_layer_1,index1 = p.pooling(sigmoid_convLayer_1, numOfFiltersLayer1)

	# -----------------------------------------------------------------------------------------------
	#                     CONVOLUTION --> SIGMOID (Activation Fn) --> POOLING (SECOND ITERATION)
	# -----------------------------------------------------------------------------------------------

	# -------------------------------------- CONVOLUTION --------------------------------------------

	convolution_layer_2 = p.convolution(array(pool_layer_1), filters2, numOfFiltersLayer2, numOfFiltersLayer1, fsize, b2)
	
	convolution_layer_2_shape = array(convolution_layer_2).shape
	# print convolution_layer_2
	# Batch normalization
	convolution_layer_2_values=[]
	for i in range(convolution_layer_2_shape[0]):
		convolution_layer_2_values.append(zscore(convolution_layer_2[0]))
	# print convolution_layer_2_valu	es[0]

	# -------------------------------------- SIGMOID ACTIVATION --------------------------------------------

	sigmoid_convLayer_2 = sigmoid(array(convolution_layer_2_values).astype(numpy.float64))

	# -------------------------------------- POOLING --------------------------------------------

	pool_layer_2,index2 = p.pooling(sigmoid_convLayer_2, numOfFiltersLayer2)

	# ---------------------------------- END OF SECOND ITERATION ---------------------------------------


	# --------------------------------------------------------------------------------------------------
	# ------------------------------[ FC --> HIDDEN LAYER --> OUTPUT ]----------------------------------
	# --------------------------------------------------------------------------------------------------


	# ----------------------------------- FULLY CONNECTED LAYER ----------------------------------------

	FC = array(pool_layer_2).ravel()

	# ---------------------------------------- HIDDEN LAYER 1 ----------------------------------------

	hidden_values1 = numpy.dot( weights_FC.T, FC) + bhl1

	sigmoid_hidden_values1 = sigmoid(hidden_values1)

	# ---------------------------------------- HIDDEN LAYER 2 ----------------------------------------

	hidden_values2 = numpy.dot( weights_HL1.T, sigmoid_hidden_values1) + bhl2

	sigmoid_hidden_values2 = sigmoid(hidden_values2)

	# ------------------------------------------- OUTPUT ----------------------------------------

	output_values = numpy.dot(weights_HL2.T, sigmoid_hidden_values2) + bFC

	output = sigmoid(output_values)
	output = output.round()
	predicted=0.

	predicted = (output[0]*2.) + (output[1]*1.)
	# print str(predicted) + ' - ' + str(target[iterat_image])
	if(predicted == target[iterat_image]):
		count += 1

		print str(output) + '==' + str(target[iterat_image]) + '---' + str(count)

	

	# if iterat_image==99:
	# 	acc=float(count)/4
	# 	print 'The System accuracy is '+ str(acc)
	# 	print acc
	# 	count=0
	# if iterat_image==199:
	# 	acc=float(count)/4
	# 	print 'The System accuracy is '+ str(acc)
		

	# 	count=0
	# if iterat_image==299:
	# 	acc=float(count)/4
	# 	print 'The System accuracy is '+ str(acc)
		

	# 	count=0
	# if iterat_image==399:
	# 	acc=float(count)/4
	# 	print 'The System accuracy is '+ str(acc)
		

	# 	count=0
acc=float(count)/4
print 'The System accuracy is '+ str(acc)
tt = time.time()-start
hours = tt/(60)
print '###################################################################'
print 'Total Time elapsed in testing the system is '+str(hours)+' Minutes!'
print ""
print "######################################################"
print "#			RESULTS				#"
print "######################################################"
print ""
print 'The System accuracy is '+ str(acc)