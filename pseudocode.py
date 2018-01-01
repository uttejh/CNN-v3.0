READ Training images
FOR EACH Image in Training Images:
	CONVERT Image into array of Pixels
	NORMALIZE the array such that mean = 0 and standard deviation = 1
INITIALISE weights and biases for all layers
START the timer
FOR EACH Epoch:
	FOR EACH Image:

		//Feed Forward
		REPEAT twice:
			CONVOLUTE each image with corresponding filters using image as input
			NORMALIZE the output
			APPLY activation function 
			APPLY Pooling 
		CONCATENATE the Pooling output to form a Fully Connected Layer
		FOR EACH upcoming layer:
			Perform DOT PRODUCT between weights and input from previous layer and ADD the corresponding bias
		ASSIGN above final output to a variable

		//Back Propagation
		CALCULATE total Loss
		FOR EACH layer:
			CALCULATE Error and Slope
			CALCULATE Change Factor by MULTIPLYING the error and slope
			CALCULATE Change in weight (deltaW) by performing DOT PRODUCT between change factor and input from the layer
			UPDATE weights by REMOVING the deltaW from each weight









