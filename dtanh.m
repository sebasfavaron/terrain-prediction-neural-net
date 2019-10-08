% Derivative of the hyperbolic tangent function in terms of itself (autonomous 
% differential equation style)
function out = dtanh(in)
  out = 1-in^2;
endfunction