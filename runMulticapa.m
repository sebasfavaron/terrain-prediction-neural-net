1;

source('multiCapa.m');
source('normalize.m');
source('graph_utils.m');

global P;
createGlobalParametersMulticapa();

#[P.inputs, maximum] = normalize(P.inputs);
tic
learn();
toc
#[P.inputs] = un_normalize(P.inputs,maximum);

graph_terrainMulticapa(P.W);
