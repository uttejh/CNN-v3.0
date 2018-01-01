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

# numOfHiddenNeurons =300
numOfOutputNeurons = 2

target = numpy.array([[0,0],[0,1],[1,0],[1,1]])


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
		for j in range(25):

			name = './new/'+str(i)+' ('+ str(j+1) +').png'
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

totalloss = []
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
count = 0

# Start the timer

target=[]
# epochs = 1000
for i in range(25):
	target.append(0.)
for i in range(25):
	target.append(1.)
for i in range(25):
	target.append(2.)
for i in range(25):
	target.append(3.)


individualcount =0
# if iterat_epoch%100 == 0:
# 	print '###############################################'
# 	print 'Output at epoch '+str(iterat_epoch)+' is:'
# 	print '###############################################'
start = time.time()
print '-------------------------------------------'
print 'The netwotk correctly predicted:'
print '-------------------------------------------'
for iterat_image in range(100):
	
	# print 'Running iteration '+ str(iterat_image)
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

	# ---------------------------------------- HIDDEN LAYER ----------------------------------------

	hidden_values = numpy.dot( weights_FC.T, FC) + bhl

	sigmoid_hidden_values = sigmoid(hidden_values)

	# ------------------------------------------- OUTPUT ----------------------------------------

	output_values = numpy.dot(weights_HL.T, sigmoid_hidden_values) + bFC

	output = sigmoid(output_values)
	# print output
	# if iterat_epoch%100 == 0:
	# 	print '---------------------------------------------------------'
	# 	print 'Output for image with label '+str(iterat_image)+' is:'
	# 	print output
	# 	print '---------------------------------------------------------'
	output = output.round()
	predicted=0.
	predicted = (output[0]*2.) + (output[1]*1.)
	# print str(predicted) + ' - ' + str(target[iterat_image])
	if(predicted == target[iterat_image]):
		count += 1
		individualcount +=1
		# print str(iterat_image+1)
		# print iterat_image
		# print str(output) + '==' + str(target[iterat_image]) + '---' + str(count)
	
	if iterat_image == 24:
		print str(individualcount) +"/25 - 0's"
		individualcount = 0
	if iterat_image == 49:
		print str(individualcount) +"/25 - 1's"
		individualcount = 0
	if iterat_image == 74:
		print str(individualcount) +"/25 - 2's"
		individualcount = 0
	if iterat_image == 99:
		print str(individualcount) +"/25 - 3's"
		individualcount = 0

	

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
acc=float(count)
print 'The System accuracy is '+ str(acc) +'%'
tt = time.time()-start
hours = tt/(60)
print '-------------------------------------------'
print '###################################################################'
print 'Total Time elapsed in testing the system is '+str(hours)+' Minutes!'
# print ""
# print "######################################################"
# print "#			RESULTS				#"
# print "######################################################"
print ""
print 'The System accuracy is '+ str(acc) +'%'
print ""

