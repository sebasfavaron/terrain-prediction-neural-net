1;

source('multilayer_perceptron.m')

function setP(Arch)
  global P;
  P.layers_vector = Arch.layers_vector;
  P.eta = Arch.eta;
  P.epsilon = Arch.epsilon;
  P.x_set = Arch.x_set;
  P.y_set = Arch.y_set;
  P.process_type = Arch.process_type;
  P.momentum = Arch.momentum;
  P.momentum_value = Arch.momentum_value;
  P.W = Arch.W;
  P.B = Arch.B;
endfunction

function printResult(Arch)
  if Arch.process_type == 0
      p_type = "batch";
  else
    p_type = "incremental";  
  endif
  printf("1: Tested [");
  printf(" %d", Arch.layers_vector);
  printf(" ] with %.2f eta, %f epsilon, in %s mode. Ended in %f seconds\n", Arch.eta, Arch.epsilon, p_type, Arch.end);
endfunction

function compare(Arch1, Arch2)
    global P;
    setP(Arch1);
    tic;
    create();
    Arch1.end = toc;
    Arch1.W = P.W;
    Arch1.B = P.B;

    printf("\n")
    
    setP(Arch2);
    tic;
    create();
    Arch2.end = toc;
    Arch2.W = P.W;
    Arch2.B = P.B;
    
    printResult(Arch1);
    printResult(Arch2);
endfunction

global P;
createGlobalParametersNN();
Arch1 = createParametersNN([2 10 1], 0.0005, 0.0001, 0, 1);
Arch2 = createParametersNN([2 10 1], 0.0004, 0.0001, 0, 1);
compare(Arch1, Arch2)