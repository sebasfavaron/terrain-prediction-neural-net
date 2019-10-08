1;

function graph_terrain(W, B, errors, epochs)
  global P;
  %[P.x_set, P.y_set, maximum] = normalize(P.x_set,P.y_set);
  x = [min(P.x_set(:,1)):0.1:max(P.x_set(:,1))]';
  y = [min(P.x_set(:,2)):0.1:max(P.x_set(:,2))]';
  [X Y] = meshgrid(x, y);
  
  for i = 1:size(X)(1)
    for j = 1:size(X)(2)
        OUT(i,j) = feed([X(i, j) Y(i,j)], W, B);
    endfor
  endfor
  for i = 1:size(P.x_set)(1)
    GUESS(i) = feed([P.x_set(:,1)'(i)  P.x_set(:,2)'(i)], W, B);
  endfor

  set(gcf, 'Position', get(0, 'Screensize'));
  
  subplot(2,2,2);
  % TODO: quiero plotear la diferencia entre el guess y el valor real pero no me viene saliendo
  #surf(P.x_set(:,1), P.x_set(:,2), GUESS);

  subplot(2,2,4)
  semilogy(epochs, errors);
  
  subplot(1,2,1);
  scatter3(P.x_set(:,1), P.x_set(:,2), P.y_set, 100, 'filled');
  colormap(summer(5));
  hold on;
  surf(X, Y, OUT);
  rotate3d on;

  drawnow;
endfunction

function out = graph_terrainMulticapa(W)
  global P;

  figure 

  x = [min(P.inputs(1, :)):0.1:max(P.inputs(1, :))];
  y = [min(P.inputs(2, :)):0.1:max(P.inputs(2, :))];
  [X Y] = meshgrid(x,y);
  
   for i = 1:size(X)(1)
    for j = 1:size(X)(2)
        [H V] = feed([X(i, j); Y(i,j)], W, P.activationFunction);
        OUT(i,j) = V{end};
    endfor
  endfor
  
  scatter3(P.inputs(1, :), P.inputs(2, :), P.outputs)
  hold on 
  surf(X, Y, OUT)
  rotate3d on;
  
endfunction

function out = graph_errors(errors)
  
  plotFigure = figure;
  semiYFigure = figure;
  semiXFigure = figure;
  logFigure = figure;

  figure(plotFigure);
  plot(errors);
  xlabel ("Epoch");
  ylabel ("Error ");
  title ("error");

  figure(semiYFigure);
  semilogy(errors);
  xlabel ("Epoch");
  ylabel ("Error (Logaritmic scale)");
  title ("semilogy error");

  figure(semiXFigure);
  semilogx(errors);
  xlabel ("Epoch (Logaritmic scale)");
  ylabel ("Error");
  title ("semilogx error");
  
  figure(logFigure);
  loglog(errors);
  xlabel ("Epoch (Logaritmic scale)");
  ylabel ("Error (Logaritmic scale)");
  title ("error");
  figure(plotFigure);

endfunction