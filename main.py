import numpy  	# Python's mathematical library
import Image	# To convert image to pixels
from procedures import *
import time		# To calculate time
import pickle	# To write and read files
import matplotlib.pyplot as plt 	# Graphical representation


numOfFiltersLayer1 = 20
numOfFiltersLayer2 = 40

# height and width of the filters
fsize = 5

alpha = 0.1

# epochs = 2000

numofInputImages = 1

numOfInputs1 = 28*28		
numOfOutputs1 = 24*24

numOfInputs2 = numOfFiltersLayer1*12*12		
numOfOutputs2 = 8*8		

numOfHiddenNeurons = 100
numOfOutputNeurons = 3

target = numpy.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])


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
		name = './dataset/'+str(i)+'.jpg'
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

# Creating filters for conv layer1
# 20*1*5*5
filters1 = p.initFilters(numOfFiltersLayer1, numOfInputs1, numOfOutputs1, fsize, 1)
# print filters1[0]
# 40*20*5*5
filters2 = p.initFilters(numOfFiltersLayer2, numOfInputs2, numOfOutputs2, fsize, numOfFiltersLayer1)
# print filters2[0]

# Initialising weights of FC Layer
weights_FC = p.initWeights(640,numOfHiddenNeurons) # FC.shape[0]=640

# Initialising weights of Hidden Layer
weights_HL = p.initWeights(numOfHiddenNeurons, numOfOutputNeurons)

# Initialise biases

# Biases of Convolution layer 1
# b1 = p.initBias(numOfFiltersLayer1)
b1=1.
# Start the timer
start = time.clock()

epochs = 1

# Start the training procedure
for iterat_epoch in range(epochs):
	for iterat_image in range(numofInputImages):

		# Read one input at a time
		input_data = imagedata[iterat_image]


		# -----------------------------------------------------------------------------------------------
		#                     CONVOLUTION --> SIGMOID (Activation Fn) --> POOLING (FIRST ITERATION)
		# -----------------------------------------------------------------------------------------------

		# -------------------------------------- CONVOLUTION --------------------------------------------

		input_data_3d = numpy.reshape(input_data, (1,28,28))
		
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

		# change biases
		convolution_layer_2 = p.convolution(array(pool_layer_1), filters2, numOfFiltersLayer2, numOfFiltersLayer1, fsize, b1)
		
		convolution_layer_2_shape = array(convolution_layer_2).shape

		# Batch normalization
		convolution_layer_2_values=[]
		for i in range(convolution_layer_2_shape[0]):
			convolution_layer_2_values.append(zscore(convolution_layer_2[0]))

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

		# ---------------------------------------- HIDDEN LAYER ----------------------------------------

		hidden_values = numpy.dot( weights_FC.T, FC) #+ bias

		sigmoid_hidden_values = sigmoid(hidden_values)

		# ------------------------------------------- OUTPUT ----------------------------------------

		output_values = numpy.dot(weights_HL.T, sigmoid_hidden_values)

		output = sigmoid(output_values)

		print output

		# ----------------------------------- END OF FORWARD PROPAGATION -----------------------------------


		# --------------------------------------------------------------------------------------------------
		# ---------------------------------------- BACK PROPAGATION ----------------------------------------
		# --------------------------------------------------------------------------------------------------

		# --------------------------------------------------------------------------------------------------
		# -------- [ CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2 <-- FC <-- HIDDEN LAYER <-- OUTPUT ] ------
		# --------------------------------------------------------------------------------------------------

		# for i in range(8):
		# 	print(target[i][0]*4) + (target[i][1]*2) + (target[i][2]*1)

		error = target[iterat_image] - output

		# ------------------------------ Hidden Layer <-- Output -----------------------------------

		slope_output_layer = derivative(output)

		# Change factor
		d_output = error*slope_output_layer

		# dw - Change in weight
		dweight_output = numpy.outer(sigmoid_hidden_values, d_output)

		# + because -*- = +
		weights_HL = weights_HL + alpha*(dweight_output)

		# -------------------------------- FC <-- Hidden Layer -----------------------------------

		slope_hidden_layer = derivative(sigmoid_hidden_values)

		error_hidden_layer = numpy.dot(weights_HL, d_output)

		d_hidden_layer = error_hidden_layer*slope_hidden_layer

		dweight_hidden = numpy.outer(FC, d_hidden_layer)

		weights_FC = weights_FC + alpha*(dweight_hidden)

		# ----------------------------- CONVOLUTION LAYER 2 <-- FC -----------------------------------

		slope_FC = derivative(FC)

		error_FC = numpy.dot( weights_FC, d_hidden_layer)

		d_FC = error_FC*slope_FC

		d_FC_2D = numpy.reshape(d_FC, (numOfFiltersLayer2, (FC.shape[0]/numOfFiltersLayer2)))

		# 40*4*4 = 40
		d_FC_sum = numpy.sum(d_FC_2D, axis=1)

		dweight_C2 = numpy.outer(d_FC_sum, pool_layer_1)

		dw_c2 = numpy.sum(dweight_C2, axis=1)

		# weight updation
		for i in range(numOfFiltersLayer2):
			w = filters2[i]

			filters2[i] = w + alpha*(dw_c2[i])			

		# --------------------- CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2 -----------------------------------

		

tt = time.clock()-start
print tt