1;

%% Global parameters. They are able to be modified mid-run
%  Users are advised AGAINST modifying the following variables:
%    layers_vector, x_set, y_set
%  They should remain static the entire run
function createGlobalParametersMulticapa()
    global P;

    BATCH_MODE = 0;
    INCREMENTAL_MODE = 1;

    #RANDOM = 1;
    #BIAS = 2;
    #ZERO = 0;
    
    fid = fopen('terrain/terrain01.data', 'r');
    fgetl(fid); % this is just to remove the 'x1 x2 z' line
    IN = fscanf(fid, '%f', [3, Inf]);
    fclose(fid);
    terrain_data = IN';
    
    % ---------------------------------------Should be static at runtime---------------------------------------
    P.hiddenLayers = [15, 30];
    P.inputs = terrain_data([1 : 350],[1 2])';
    P.outputs = terrain_data([1 : 350],3)';

    % ---------------------------------------To modify/access at runtime---------------------------------------
    P.eta = 0.01; % Learning rate
    P.epsilon = 0.001;
    P.process_type = BATCH_MODE;
    
    P.variable_eta_flag = false;
    P.variable_eta_delta = 0.1;
    P.momentum = false;
    P.variable_eta_n = 0.05;
    v_eta_min_value = 0.001;
    v_eta_decrease_amount = 0.000002;
    % Function has a minimum value greater than 0, otherwise it would stop learning
    P.variable_eta = @(x)((x > v_eta_min_value)*(x - v_eta_decrease_amount) + (x <= v_eta_min_value)*(v_eta_min_value));
    
    P.activationFunction = @(x) (tanh(x)); % tanjh function can be in an other file
    %P.activationFunction = @(x)(1/(1+exp(-x))); % sigmoid function can be in an other file
    
    P.activationFunctionDeriv = @(x)((1 - tanh(x)^2)); % tanh function can be in an other file
    %P.activationFunctionDeriv = @(x)(P.activationFunction(x) * (1 - P.activationFunction(x)));% sigmoid function can be in an other file
    
    %P.momentum = 0; % Fraction of previous delta weights to update current weight (wi = wi + momentum*wi-1)
    %P.W = generateAllLayerWeights(size(P.inputs)(1), P.hiddenLayers, size(P.outputs)(1));
%    P.W = load('startW_arq4');
    load('startW_arq4');
    P.W = startW;
    P.dW = generateAllLayerWeightsZero(size(P.inputs)(1), P.hiddenLayers, size(P.outputs)(1));
    
    %---------------------------------------Outputs ----------------------------------------------------------------
    P.errors = [];
    P.etas = [];
    P.epochs = 0;
    
endfunction

% Splits terrain_data into P.training_data and P.testing_data according to P.training_select_set
% 	Any terrain_data(i,:) where i belongs to P.training_select_set will go to P.training_data
%	The remainder of terrain_data goes to P.testing_data
function split_data(terrain_data)
	global P;
	terrain_data_len = length(terrain_data);

	% Make training_set
	for i = 1:length(P.training_select_set)
		select_idx = P.training_select_set(i);
		P.training_data = [P.training_data; terrain_data(select_idx, :)];
	endfor

	% Make testing_set
	P.testing_select_set = setdiff([1:terrain_data_len], P.training_select_set);
    P.testing_data = [];
	for i = 1:length(P.testing_select_set)
		select_idx = P.testing_select_set(i);
		P.testing_data = [P.testing_data; terrain_data(select_idx, :)];
	endfor
endfunction

%% Global parameters. They are able to be modified mid-run
%  Users are advised AGAINST modifying the following variables:
%    layers_vector, x_set, y_set
%  They should remain static the entire run
function createGlobalParametersNN()
    global P;

    BATCH_MODE = 0;
    INCREMENTAL_MODE = 1;

    PERCENT = 0;
    SELECT_SET = 1;

    RANDOM = 1;
    BIAS = 2;
    ZERO = 0;

	fid = fopen('terrain/terrain01.data', 'r');
    fgetl(fid); % this is just to remove the 'x1 x2 z' line
    IN = fscanf(fid, '%f', [3, Inf]);
    fclose(fid);
    terrain_data = IN';
	terrain_data_len = length(terrain_data);
    
    % ---------------------------------------To modify---------------------------------------
    P.layers_vector = [2 15 15 1];
    P.eta = 0.003; % Learning rate
    P.epsilon = 0.0005;
    P.process_type = BATCH_MODE;
    P.variable_eta_flag = false;
    v_eta_min_value = 0.0007;
    v_eta_decrease_amount = 0.00000002;
    P.momentum_flag = true;
    P.momentum = 0.01; % Fraction of previous delta weights to update current weight (wi = wi + momentum*wi-1)
    P.training_mode_flag = SELECT_SET;
	% Only active if flag set to PERCENT
    P.training_percent = 0.9;
    % Create a vector with any amount of numbers from 1 to terrain_data_len
	% Only active if flag set to SELECT_SET
    P.training_select_set = [1:300 400:terrain_data_len]; %ex: all numbers but 301-399 range
    
    % ---------------------------------------To read-only---------------------------------------
	if(P.training_mode_flag == PERCENT)
		% modify P.training_select_set and use that for both training modes
		training_amount = floor(P.training_percent * length(terrain_data));
		P.training_select_set = randperm(terrain_data_len, training_amount);
	endif
	P.training_data = [];
	P.testing_data 	= [];
	split_data(terrain_data);

    P.x_set = P.training_data(:,[1 2]);
    P.y_set = P.training_data(:,3);
    P.variable_eta = @(x)((x > v_eta_min_value)*(x - v_eta_decrease_amount) + (x <= v_eta_min_value)*(v_eta_min_value));
    P.momentum_value = createArrays(P.layers_vector, ZERO);
    P.W = createArrays(P.layers_vector, RANDOM);
    P.B = createArrays(P.layers_vector, BIAS);

endfunction

%% This parameters will not be global. They are just used for NN comparisons
function [Param] = createParametersNN(layers_vector, eta, epsilon, process_type, momentum)
    
    BATCH_MODE = 0;
    INCREMENTAL_MODE = 1;

    RANDOM = 1;
    BIAS = 2;
    ZERO = 0;
    
    fid = fopen('terrain/terrain01.data', 'r');
    fgetl(fid); % this is just to remove the 'x1 x2 z' line
    IN = fscanf(fid, '%f', [3, Inf]);
    fclose(fid);
    terrain_data = IN';
    
    % Should be static at runtime
    Param.layers_vector = layers_vector;
    Param.x_set = terrain_data(:,[1 2]);%[ 1 1; -1 -1; 1 -1; -1 1 ];#terrain_data(:,[1 2]);
    Param.y_set = terrain_data(:,3);%[ -1; -1; 1; 1 ];#terrain_data(:,3);

    % To modify/access at runtime
    Param.eta = eta; % Learning rate
    Param.epsilon = epsilon;
    Param.process_type = process_type;
    Param.momentum = 0;
    Param.momentum_value = createArrays(Param.layers_vector, ZERO);
    Param.W = createArrays(Param.layers_vector, RANDOM);
    Param.B = createArrays(Param.layers_vector, BIAS);
endfunction


% generate all layers weights matrix
% inputSize is the amount of input nodes
% hiddenLayersSize is an array of the amount of nodes in each hidden layer
% outputSize is the amount of output nodes
% returns a cell array of weights
function W = generateAllLayerWeights(inputSize, hiddenLayersSize, outputSize)
    % TODO: Validate if hiddenLayers is not empty
    widx = 1;
    W{widx} = generateWeights(inputSize, hiddenLayersSize(1));
  
    for i = 1 : size(hiddenLayersSize)(2) - 1
      widx = widx + 1;
      W{widx} = generateWeights(hiddenLayersSize(i), hiddenLayersSize(i + 1));
    end
  
    W{widx + 1} = generateWeights(hiddenLayersSize(size(hiddenLayersSize)(2)), outputSize);
  end


function W = generateAllLayerWeightsZero(inputSize, hiddenLayersSize, outputSize)
    % TODO: Validate if hiddenLayers is not empty
    widx = 1;
    W{widx} = zeros(hiddenLayersSize(1), inputSize + 1);
  
    for i = 1 : size(hiddenLayersSize)(2) - 1
      widx = widx + 1;
      W{widx} = zeros(hiddenLayersSize(i + 1), hiddenLayersSize(i) + 1);
    end
  
    W{widx + 1} = zeros(outputSize, hiddenLayersSize(size(hiddenLayersSize)(2)) + 1);
  end



function result = createArrays(arraySize, type)
    RANDOM = 1;
    BIAS = 2;
    ZERO = 0;
    % arraySize es un vector con las dimensiones de cada capa ej: [4,3,2, 3, 5]
    % nArrays es el numero de capas a crear
    n_arrays = size(arraySize)(2) - 1;
    result = cell(1, n_arrays );
    for i = 1 : (n_arrays)
        switch (type)
            case (RANDOM)
                result{i} = randn([arraySize(i), arraySize(i + 1)]);
            case (BIAS)
                result{i} = (zeros([1, arraySize(i+1)]) - 1);
            otherwise
                
                result{i} = (zeros([arraySize(i), arraySize(i + 1)]));
        endswitch
    end
end

% generate random weights for each layer
% Msize is current layer nodes amount (without bias)
% M1size is the next layer nodes amount (without bias)
function W = generateWeights(Msize, M1size)
  W = rand(M1size, Msize + 1) - 0.5;
end

