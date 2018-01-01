def ReadImages:
	for image in range(TrainingImages):
		image = convertToPixels(image)
		image = Zscore(image)
	return image	

filters1 = initWeights()
filters2 = initWeights()

weights_HL = initWeights()
weights_FC = initWeights()

bias1 = initBias()
bias2 = initBias()
biasFC = initBias()
biasHL = initBias()

def Zscore:
	return (x-mean)/std_deviation

def main:
	for epoch in range(epochs):
		for image in range(TrainingImages):
			# feed forward
			input = ReadImages()

			c = convolute(input)
			n = normalize(c)
			act_fn = sigmoid(n)
			pool = pool(act_fn)

			c = convolute(pool)
			n = normalize(c)
			act_fn = sigmoid(n)
			pool = pool(act_fn)

			FC = Flatten(pool)

			hl = dot(weights_FC, FC)

			output = dot(weights_HL, hl)

			output = sigmoid(ouput)

			# Back propagation
			error = target - output

			for each layer:
				slope = derivative(x)

				change_factor  = error*slope

				dw = dot(change_factor, values)

				w = w - dw
				


