1;

source('multilayer_perceptron.m')
source('normalize.m')
source('graph_utils.m')

global P;
createGlobalParametersNN();

#[P.x_set, maximum] = normalize(P.x_set);
tic
create();
toc
#[P.x_set] = un_normalize(P.x_set,maximum);

#graph_terrain(P.W, P.B);

