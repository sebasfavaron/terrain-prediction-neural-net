1;

source('graph_terrain.m')
source('multiCapa.m');
source('normalize.m');

global P;
close all;


model3d = figure;

x = [min(P.inputs(1, :)):0.1:max(P.inputs(1, :))];
y = [min(P.inputs(2, :)):0.1:max(P.inputs(2, :))];
[X Y] = meshgrid(x,y);

for i = 1:size(X)(1)
for j = 1:size(X)(2)
	[H V] = feed([X(i, j); Y(i,j)], P.W, P.activationFunction);
	OUT(i,j) = V{end};
endfor
endfor

scatter3(P.inputs(1, :), P.inputs(2, :), P.outputs)
hold on 
surf(X, Y, OUT)
rotate3d on;
print('3dmodel.png');



plotFigure = figure;
semiYFigure = figure;
semiXFigure = figure;
logFigure = figure;

figure(plotFigure);
plot(P.errors);
xlabel ("Epoch");
ylabel ("Error ");
title ("error");
print('error.png');

figure(semiYFigure);
semilogy(P.errors);
xlabel ("Epoch");
ylabel ("Error (Logaritmic scale)");
title ("semilogy error");
print('semilogy.png');

figure(semiXFigure);
semilogx(P.errors);
xlabel ("Epoch (Logaritmic scale)");
ylabel ("Error");
title ("semilogx error");
print('semilogx.png');

figure(logFigure);
loglog(P.errors);
xlabel ("Epoch (Logaritmic scale)");
ylabel ("Error (Logaritmic scale)");
title ("Logaritmic error");
print('Logaritmic.png');


if(P.variable_eta_flag)
	etaFigure = figure;
	figure(etaFigure);
	plot(P.etas);
	xlabel ("Iterations");
	ylabel ("Eta");
	title ("Eta");
	print('etas.png');
end
