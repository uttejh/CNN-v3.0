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

# epochs = 2000

numofInputImages = 4

numOfInputs1 = 28*28		
numOfOutputs1 = 24*24

numOfInputs2 = numOfFiltersLayer1*12*12		
numOfOutputs2 = 8*8		

numOfHiddenNeurons = 200
numOfOutputNeurons = 2

target = numpy.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])


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
		
		# im = numpy.roll(im,-1,axis=0) # expand up

		# im = numpy.roll(im,1,axis=0) # expand down

		# im = numpy.roll(im,2,axis=1) # expand right

		# im = numpy.roll(im,-2,axis=1) # expand left
			

		# Normalizing data so that each column of Z has mean 0 and standard 1
		# also called as Z-score normalization
		im = zscore(im)

		data.append(im)
	
	return data

start = time.time()
imagedata = readAllImages() # array of pixel data of all the input data 
tt = time.time() - start
print ''
print 'Time taken to read and normalize the training dataset is: '+str(tt)

p = Procedures()

filters1 = []
filters2 = []

err_hl = []
err_FC = []
err_c2 = []

# # Creating filters for conv layer1
# # 20*1*5*5
# filters1 = p.initFilters(numOfFiltersLayer1, numOfInputs1, numOfOutputs1, fsize, 1)
# # print filters1[0]
# # 40*20*5*5
# filters2 = p.initFilters(numOfFiltersLayer2, numOfInputs2, numOfOutputs2, fsize, numOfFiltersLayer1)
# # print filters2[0]

# # Initialising weights of FC Layer
# weights_FC = p.initWeights(640,numOfHiddenNeurons) # FC.shape[0]=640

# # Initialising weights of Hidden Layer
# weights_HL = p.initWeights(numOfHiddenNeurons, numOfOutputNeurons)

# # Initialise biases

# # Biases of Convolution layer 1
# b1 = p.initBias(numOfFiltersLayer1)
# b2 = p.initBias(numOfFiltersLayer2)

# bhl = p.initBias(1)
# bFC = p.initBias(1)

f = open('./weights/filters1.txt')
filters1 = pickle.load(f)
f.close()

f = open('./weights/filters2.txt')
filters2 = pickle.load(f)
f.close()

f = open('./weights/FC_to_HL.txt')
weights_FC = pickle.load(f)
f.close()

f = open('./weights/HL_to_output.txt')
weights_HL = pickle.load(f)
f.close()

f = open('./weights/b1.txt')
b1 = pickle.load(f)
f.close()

f = open('./weights/b2.txt')
b2 = pickle.load(f)
f.close()

f = open('./weights/bhl.txt')
bhl = pickle.load(f)
f.close()

f = open('./weights/bFC.txt')
bFC = pickle.load(f)
f.close()

totalloss = []
tt1=0
tt2=0

tt3=0
tt4=0
tt5=0
tt6=0
tt7=0
tt8 =0


# Start the timer
start = time.time()

epochs = 1

# Start the training procedure
for iterat_epoch in range(epochs):
	# print 'Running epoch: ' + str(iterat_epoch) + ' ....'

	for iterat_image in range(numofInputImages):

		# Read one input at a time
		input_data = imagedata[iterat_image]


		# -----------------------------------------------------------------------------------------------
		#                     CONVOLUTION --> SIGMOID (Activation Fn) --> POOLING (FIRST ITERATION)
		# -----------------------------------------------------------------------------------------------

		# -------------------------------------- CONVOLUTION --------------------------------------------

		input_data_3d = numpy.reshape(input_data, (1,28,28))
		# print input_data_3d
		start = time.time()
		convolution_layer_1 = p.convolution(input_data_3d, filters1, numOfFiltersLayer1, 1, fsize, b1)		
		tt1 += time.time() - start


		convolution_layer_1_shape = array(convolution_layer_1).shape

		# Batch normalization
		convolution_layer_1_values=[]
		for i in range(convolution_layer_1_shape[0]):
			convolution_layer_1_values.append(zscore(convolution_layer_1[0]))

		# -------------------------------------- SIGMOID ACTIVATION --------------------------------------------

		sigmoid_convLayer_1 = sigmoid(array(convolution_layer_1_values))

		# -------------------------------------- POOLING --------------------------------------------

		start = time.time()
		pool_layer_1,index1 = p.pooling(sigmoid_convLayer_1, numOfFiltersLayer1)
		tt2 += time.time() - start

		# -----------------------------------------------------------------------------------------------
		#                     CONVOLUTION --> SIGMOID (Activation Fn) --> POOLING (SECOND ITERATION)
		# -----------------------------------------------------------------------------------------------

		# -------------------------------------- CONVOLUTION --------------------------------------------
		start = time.time()

		convolution_layer_2 = p.convolution(array(pool_layer_1), filters2, numOfFiltersLayer2, numOfFiltersLayer1, fsize, b2)
		tt3 += time.time() - start
		
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
		start = time.time()

		pool_layer_2,index2 = p.pooling(sigmoid_convLayer_2, numOfFiltersLayer2)
		tt4 += time.time() - start

		# ---------------------------------- END OF SECOND ITERATION ---------------------------------------


		# --------------------------------------------------------------------------------------------------
		# ------------------------------[ FC --> HIDDEN LAYER --> OUTPUT ]----------------------------------
		# --------------------------------------------------------------------------------------------------


		# ----------------------------------- FULLY CONNECTED LAYER ----------------------------------------

		FC = array(pool_layer_2).ravel()

		# ---------------------------------------- HIDDEN LAYER ----------------------------------------

		start = time.time()

		hidden_values = numpy.dot( weights_FC.T, FC) + bhl

		sigmoid_hidden_values = sigmoid(hidden_values)
		tt5 += time.time() - start

		# ------------------------------------------- OUTPUT ----------------------------------------

		output_values = numpy.dot(weights_HL.T, sigmoid_hidden_values) + bFC
		
		start = time.time()

		output = sigmoid(output_values)
		tt6 += time.time() - start

		# if iterat_epoch%100 == 0:
		# 	print '---------------------------------------------------------'
		# 	print 'Output for image with label '+str(iterat_image)+' is:'
		# 	print output
		# 	print '---------------------------------------------------------'
		# if iterat_epoch == epochs-1:
		# 	print '---------------------------------------------------------'
		# 	print 'Final Output for image with label '+str(iterat_image)+' is:'
		# 	print output
		# 	print '---------------------------------------------------------'

		# ----------------------------------- END OF FORWARD PROPAGATION -----------------------------------


		# --------------------------------------------------------------------------------------------------
		# ---------------------------------------- BACK PROPAGATION ----------------------------------------
		# --------------------------------------------------------------------------------------------------

		# --------------------------------------------------------------------------------------------------
		# -------- [ CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2 <-- FC <-- HIDDEN LAYER <-- OUTPUT ] ------
		# --------------------------------------------------------------------------------------------------

		# for i in range(8):
		# 	print(target[i][0]*4) + (target[i][1]*2) + (target[i][2]*1)
		start = time.time()

		error = target[iterat_image] - output

		loss = 0.5*error**2
		totalloss.append(numpy.sum(loss))


		# ------------------------------ Hidden Layer <-- Output -----------------------------------

		slope_output_layer = derivative(output)

		# Change factor
		d_output = error*slope_output_layer

		# dw - Change in weight
		dweight_output = numpy.outer(sigmoid_hidden_values, d_output)

		# + because -*- = +
		weights_HL = weights_HL + alpha*(dweight_output)

		bhl += alpha*numpy.sum(d_output) 

		# -------------------------------- FC <-- Hidden Layer -----------------------------------

		slope_hidden_layer = derivative(sigmoid_hidden_values)

		error_hidden_layer = numpy.dot(weights_HL, d_output)

		err_hl.append(numpy.sum(0.5*error_hidden_layer**2))

		d_hidden_layer = error_hidden_layer*slope_hidden_layer

		dweight_hidden = numpy.outer(FC, d_hidden_layer)

		weights_FC = weights_FC + alpha*(dweight_hidden)

		bFC += alpha*numpy.sum(d_hidden_layer)

		# ----------------------------- CONVOLUTION LAYER 2 <-- FC -----------------------------------

		slope_FC = derivative(FC)

		error_FC = numpy.dot( weights_FC, d_hidden_layer)

		err_FC.append(numpy.sum(0.5*error_FC**2))

		d_FC = error_FC*slope_FC

		d_FC_3D = numpy.reshape(d_FC, (numOfFiltersLayer2, 4,4))

		d_FC_2D = numpy.reshape(d_FC, (numOfFiltersLayer2, 4*4))

		index2_reshape = numpy.reshape(index2, (numOfFiltersLayer2,4*4))	

		d_FC_new = []
		for i in range(numOfFiltersLayer2):
			scalar_dw = numpy.outer(d_FC_3D[i], pool_layer_1)
			dw_c2 = numpy.sum(scalar_dw)

			w = filters2[i]

			# Weight updation
			filters2[i] = w + alpha*(dw_c2)	

			b2[i] += alpha*numpy.sum(d_FC_2D[i])

			tomodify = numpy.zeros((8*8))
			xx = index2_reshape[i].astype(int)
			yy = d_FC_2D[i]

			for (ind, rep) in zip(xx, yy):
					tomodify[ind] = rep	
			d_FC_new.append(tomodify)

		d_FC_new_reshape = numpy.reshape(d_FC_new, (40,8,8))

		# --------------------- CONVOLUTION LAYER 1 <-- CONVOLUTION LAYER 2 -----------------------------------
		
		errr=[]
		for n2 in range(numOfFiltersLayer2):
			err=[]
			for n1 in range(numOfFiltersLayer1):
				new = numpy.zeros((12,12))
				for ii in range(8):
					for jj in range(8):
						for k in range(5):
							for l in range(5):
								new[ii+k][jj+l] += d_FC_new_reshape[n2][ii][jj] * filters2[n2][n1][k][l]
				err.append(new)
			errr.append(err)

		slope_conv2 = derivative(array(pool_layer_1))

		error_conv2 = numpy.sum(errr,axis=0)

		err_c2.append(numpy.sum(0.5*error_conv2**2))

		d_c2 = error_conv2*slope_conv2
		
		for i in range(numOfFiltersLayer1):
			scalar_dw_c1 = numpy.outer(d_c2[i], input_data)

			dw_c1 = numpy.sum(scalar_dw_c1)

			w_c1 = filters1[i]

			# Weight updation
			filters1[i] = w_c1 + alpha*(dw_c1)	

			b1[i] += alpha*numpy.sum(d_c2[i])
		tt8 += time.time() - start
		

tt7 = time.time()-start
hours = tt/(3600)
print ''
print 'Total Time elapsed in training the system is '+str(tt7)+' seconds!'
print ''
print ''

print 'The average time taken for Convolution module is '+str(tt1/4)+' seconds!'
print 'The output shape is ' + str(convolution_layer_1_shape)
print ''
print ''

print 'The average time taken for Pooling module is '+str(tt2/4)+' seconds!'
print 'The output shape is ' + str(array(pool_layer_1).shape)
print ''
print ''

print 'The average time taken for Convolution module is '+str(tt3/4)+' seconds!'
print 'The output shape is ' + str(convolution_layer_2_shape)
print ''
print ''

print 'The average time taken for Pooling module is '+str(tt4/4)+' seconds!'
print 'The output shape is ' + str(array(pool_layer_2).shape)
print ''
print ''

print 'The average time taken at FC layer '+str(tt5/4)+' seconds!'
print 'The output shape is ' + str(array(FC).shape)
print ''
print ''

print 'The average time taken at Hidden layer '+str(tt6/4)+' seconds!'
print 'The output shape is ' + str(array(hidden_values).shape)
print ''
print ''

print 'The average time taken for Back-Propagation is '+str(tt8/4)+' seconds!'

print ''
print ''
