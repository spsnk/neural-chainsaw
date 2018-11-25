function [valerror] = epoch_training(v1, v2)
%Esta Funcion ejecuta una época propagando los valores hacia adelante y
%realizando backpropagation, regresando el error de epoca.

a0 = P';

a1 = logsig (W1 * a0 + b1);

a2 = purelin(W2 * a1 + b2);

a = a2;

e = T - a;

valerror = sum(e, 'all') / length(T);