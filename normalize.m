1;

function [x,maximum] = normalize(x)
  maximum = max(max(abs(x)));
  #maximum_x = max(max(abs(x(:,1))));
  #maximum_y = max(max(abs(x(:,2))));
  x = x ./ maximum;
endfunction

function [x] = un_normalize(x, maximum)
  x = x .* maximum;
endfunction