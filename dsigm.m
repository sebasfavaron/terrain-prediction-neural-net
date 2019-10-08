% Derivative of the sigmoid function in terms of itself (autonomous differential
% equation style)
function out = dsigm(in)
  out = in*(1-in);
endfunction