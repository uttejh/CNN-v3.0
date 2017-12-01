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

numofInputImages = 8

numOfInputs1 = 28*28		
numOfOutputs1 = 24*24

numOfInputs2 = numOfFiltersLayer1*12*12		
numOfOutputs2 = 8*8		

# Reads and converts the input images into array of pixels
def readAllImages():
	data = []
	for i in range(numofInputImages):
		name = './dataset/'+str(i)+'.jpg'
		image = Image.open(name)

		im = numpy.array( image, dtype="double" ) 
		data.append(im)

	return data

imagedata = readAllImages() # array of pixel data of all the input data 


p = Procedures()

filters1 = []
filters2 = []

# Creating filters for conv layer1
filters1 = p.initFilters(numOfFiltersLayer1, numOfInputs1, numOfOutputs1, fsize, 1)

filters2 = p.initFilters(numOfFiltersLayer2, numOfInputs2, numOfOutputs2, fsize, numOfFiltersLayer1)

# Start the timer
start = time.clock()

# Start the training procedure
for iterat_epoch in range(epochs):
	for iterat_image in range(numofInputImages):

		# Read one input at a time
		input_data = imagedata[iterat_image]


		# -----------------------------------------------------------------------------------------------
		#                     CONVOLUTION --> SIGMOID (Activation Fn) --> POOLING (FIRST ITERATION)
		# -----------------------------------------------------------------------------------------------

		# ---------------------------------------CONVOLUTION---------------------------------------------

		
