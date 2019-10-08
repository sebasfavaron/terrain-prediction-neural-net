function retval = perceptron(input, weights, gamma, expectedValue)
	h = weights*input

	output = 1/(1+exp(-h))

	deltaWeights = (gamma*(expectedValue - output)*input)'

	retval = weights+deltaWeights
endfunction