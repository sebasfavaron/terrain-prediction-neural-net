% input = zeros(3,1);
% input(1,1) = -1;
% expectedValue = 0;







inputs = [ -1,0,0,0,0 ; -1,0,0,0,1 ; -1,0,0,1,1 ; -1,0,1,0,0 ; -1,0,1,0,1 ; -1,0,1,1,0 ; -1,0,1,1,1 ; -1,1,0,0,0 ; -1,1,0,0,1 ; -1,1,0,1,0 ; -1,1,0,1,1 ; -1,1,1,0,0 ; -1,1,1,0,1 ; -1,1,1,1,0 ; -1,1,1,1,1  ; -1,0,0,1,0]
expectedValues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0];

N = size(inputs)(2)
weights = rand(1,N)-0.5
gamma = 1.25;

for j = 1:30
	for i = 1:size(inputs)(1) - 1
		input = inputs(i,:)' 
		expectedValue = expectedValues(i)
		weights = perceptron(input,weights,gamma,expectedValue)
	endfor
endfor


sigmoid(weights*(inputs(1,:)'))
sigmoid(weights*(inputs(2,:)'))
sigmoid(weights*(inputs(15,:)'))
sigmoid(weights*(inputs(16,:)'))
