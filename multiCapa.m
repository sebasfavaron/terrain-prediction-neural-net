
source('Parameters.m');

% add bias to V
function VwithBias = addBiasTo(V)
  Vsize = size(V)(1);
  VwithBias = ones(Vsize + 1, 1) * -1;
  VwithBias(2:Vsize+1, 1) = V(:,1);
end

function [H, V2] = feedLayer(V, W, activationFunction)
  VwithBias = addBiasTo(V);
  H = W * VwithBias;
  V2 = arrayfun(activationFunction,H);
end

function [H, V] = feed(E, W, activationFunction)
  V{1} = E;
  % i = 1;
  for i = 1 : size(W)(2) - 1
    [H{i+1}, V{i+1}] = feedLayer(V{i}, W{i}, activationFunction);
  end
  % i++;
  i = size(W)(2);
  ident = @(x) x;
  [H{i+1}, V{i+1}] = feedLayer(V{i}, W{i}, ident);
end

% First backpropagation iteration, which starts from the output of the net
%
% Parameters:
% V: Column vector, where each element contains the value of the node
% H: Column vector, where each element contains the value of the node which has been applied the acitvation function
% S: Column vector, where each element contains the expected output of the net
% activationFunctionDeriv: Derivate of the activation function
%
% Output:
% d: Column vector, which represents deltas to be applied to the weights
function d = initialBackPropagation(H, V, S, activationFunctionDeriv)
  H1 = arrayfun(activationFunctionDeriv, H); % arrayfun() applies the function to each element of the array
  d = H1 .* (S - V);
end

function d = layerBackPropagation(H, W, prevD, activationFunctionDeriv)
  H1 = arrayfun(activationFunctionDeriv, H);
  transW = W(:,2:size(W)(2))';
  d = H1 .* (transW * prevD);
end

% Function that calculates all deltas to be added to the weights
% 
% Parameters:
% V: Array of column vectors, where each vector represents a layer and each element of the vector is the value of the node
% H: Same as H but each element of the vector has been applied the activation function
% W: Array of matrices, where each matrix reprenests the weights from one layer to another
% S: Column vector with expected output values
% activationFunctionDeriv: derivate of the activation function
function d = getAlld(H, V, W, S, activationFunctionDeriv)
  iterations = size(H)(2);
  d{iterations} = initialBackPropagation(H{iterations}, V{iterations}, S, @(x) 1);
  for i = iterations - 1 : -1 : 2
    d{i} = layerBackPropagation(H{i}, W{i}, d{i+1}, activationFunctionDeriv);
  end
end

function [W dW] = backPropagation(H, V, W, S, learningFactor, activationFunctionDeriv, lastdW)
  global P;
  d = getAlld(H, V, W, S, activationFunctionDeriv);

  for i = 1 : size(W)(2)
    dW{i} = learningFactor * d{i+1} * addBiasTo(V{i})';
    if(P.momentum)
      W{i} = W{i} + dW{i} + 0.9 * lastdW{i};
    else
      W{i} = W{i} + dW{i};
    end
  end
end

function [newEta, newDescending] = adjustEta(errors, n, deltaEta, eta,descending)
  global P;
  len = length(errors);
  if(len > 1)
     prev = errors(len-1);
     actual = errors(len);
    if(actual < prev)
      if(descending+1 == 3)
        newEta = eta+n;
        newDescending = 0;
      else
        newDescending = descending+1;
        newEta = eta;
      end
    else
      newDescending = 0;
      newEta = eta*(1-deltaEta);
    end
    
  else
    newEta = eta;
    newDescending= descending;
  end
  P.etas = [P.etas;newEta];
end

function learn()
  global P;
  accumError = 1;
  descending =0;
  
  close all;
  
  screensize = get( groot, 'Screensize' );
  hSize = screensize(3);
  vSize = screensize(4);
  if(P.variable_eta_flag)
    logErrorFigure = figure('Position', [0 1000 (hSize/2)-20 vSize/3]);
    varEtaFigure   = figure('Position', [0 0    (hSize/2)-20 vSize/3]);
    t = [];
    etas = [];
  else
    logErrorFigure = figure('Position', [hSize/20 vSize/10 hSize/3 vSize/2]);
  endif

  while accumError > P.epsilon && P.epochs < 500
    accumError = 0;
    [P.inputs, P.outputs] = shuffleInputs(P.inputs, P.outputs);
    for i = 1 : size(P.inputs)(2)
      [H, V] = feed(P.inputs(:,i), P.W, P.activationFunction);
      [P.W, P.dW] = backPropagation(H, V, P.W, P.outputs(:,i), P.eta, P.activationFunctionDeriv, P.dW);
      accumError = accumError + mean( (V{size(V)(2)} - P.outputs(:,i)) .^2 );
      if(accumError == Inf)
        return;
      end
    end
   

    accumError = accumError / size(P.inputs)(2);
    P.epochs++;
    fprintf("Error is: %d epoch:%d \n", accumError, P.epochs);
    P.errors =[P.errors;accumError];
    figure(logErrorFigure),
    loglog(P.errors);
    xlabel ("Iterations (Logaritmic scale)");
    ylabel ("Error (Logaritmic scale)");
    title ("Error");

    if(P.variable_eta_flag)
      [P.eta, descending] = adjustEta(P.errors,P.variable_eta_n, P.variable_eta_delta, P.eta,descending);
      etas = [etas P.eta];
      t = [t P.epochs];
      figure(varEtaFigure), plot(t, etas);
      xlabel ("Iterations");
      ylabel ("Eta");
      title ("Eta");
    end
    drawnow;
    
  end

end

function S = test(inputs, W, activationFunction)
  for i = 1 : size(inputs)(2)
    [H, V] = feed(inputs(:,i), W, activationFunction);
    S{i} = V{size(V)(2)};
  end
end


function [shufInputs, shufOutputs] = shuffleInputs(inputs, outputs)
  x = inputs(1,:);
  y = inputs(2,:);
  z = outputs;
  ind = randperm (numel (x));
  x = x(ind);
  y = y(ind);
  z = z(ind);

  shufInputs = [x ; y];
  shufOutputs = z;

end
  

