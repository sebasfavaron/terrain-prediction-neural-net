1;

source('Parameters.m')

% w_vector is a vector of weight matrices
% b_vector is a vector of biases vectors

function a = feed(input_x , w_vector, b_vector)

    x_size = size(input_x)(1);
    w_size = size(w_vector)(2);
    a = input_x;
    for i = 1 : w_size
        h = (a * w_vector{i}) + generate_bias_matrix(b_vector{i}, x_size);
        y = tanh(h);
        a = y;
    end
endfunction

function a_vector = internal_feed(input_x , w_vector, b_vector, layers_vector)
    x_size = size(input_x)(1);
    w_size = size(w_vector)(2);

    a_vector =  cell(1, w_size + 1);
    for i = 1 : (w_size + 1)
        a_vector{i} = zeros([x_size, w_size]);
    end

    a_vector{1} = input_x;
    for i = 1 : w_size
        h = (a_vector{i} * w_vector{i}) + generate_bias_matrix(b_vector{i}, x_size);
        y = tanh(h);
        a_vector{i+1} = y;
    end
endfunction

function b_matrix = generate_bias_matrix(b_vector, amount_of_patterns)
    b_matrix = b_vector;
    if(amount_of_patterns > 1)
        for  i = 2:amount_of_patterns
            b_matrix = [b_matrix; b_vector];
        end
    endif
endfunction

function error = epoch_error(o_set, y_set)
    error= mean((o_set - y_set).^2);
endfunction

function delta = delta_weight(prev_delta, W, A)
    % derivative tanh
    f_prime = 1 - A .* A;

    delta = f_prime .* (prev_delta * W');
endfunction

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
endfunction

function [w_vector, b_vector] = back_propagate(w_vector, b_vector, layers_vector, y_set, y_hat, eta, a_vector, input)
    global P;
    
    RANDOM = 1;
    BIAS = 2;
    ZERO = 0;
    layers_size = size(w_vector)(2);
    delta = y_set - y_hat;
    f_prime = 1 - (y_hat .* y_hat);
    delta = f_prime .* delta;
    deltas = createArrays(layers_vector, ZERO);
    
    deltas{layers_size} = delta;
    for i = layers_size - 1 : -1 : 1
        deltas{i} = delta_weight(deltas{i + 1}, w_vector{i+1}, a_vector{i + 1});
    end

    w_vector{1} = w_vector{1} - eta * (input') * deltas{1};
    b_vector{1} = b_vector{1} - eta * sum(deltas{1});
    for i = 1 : layers_size
        w_vector{i} = w_vector{i} + eta * (a_vector{i}') * deltas{i} + P.momentum_value{i}; 
        b_vector{i} = b_vector{i} + eta * sum(deltas{i});
        if(P.momentum_flag)
            #P.momentum_value{i} = P.momentum_value{i} + P.momentum * w_vector{i};
        endif
    end
    
endfunction

function fit()
    global P;

    % tenemos que calcular primero la de la ultima capa
    a_vector = internal_feed(P.x_set, P.W, P.B, P.layers_vector);
    error = epoch_error(P.y_set, a_vector{end})
    epoch = 0;
    prev_eta = P.eta;
    errors = [];
    epochs = [];
    while(error > P.epsilon)
        
        [P.W, P.B] = back_propagate(P.W, P.B, P.layers_vector, P.y_set, a_vector{end}, P.eta, a_vector, P.x_set);
        a_vector = internal_feed(P.x_set, P.W, P.B, P.layers_vector);

        error = epoch_error(P.y_set, a_vector{end});
        if(mod(epoch, 1000) == 0)
            error
            hold off;
            errors = [errors, error];
            epochs = [epochs, epoch];
            graph_terrain(P.W, P.B, errors, epochs);
        endif

        if(P.variable_eta_flag)
            P.eta = P.variable_eta(P.eta);
        endif
        
        epoch = epoch + 1;
    end
    
    printf("%d epochs \t %f error (MSE). \n", epoch, error);

endfunction

function ctrlC()
    global P;
    #graph_terrain(P.W, P.B);
endfunction

%% Creates and trains neural network with input parameters
function create()
    cleanupObj = onCleanup(@ctrlC);

    fit();

endfunction
