function [valerror] = epoch_validation(P, T, W1, b1, W2, b2)
%Esta funcion ejecuta la validacion propagando los valores hacia adelante y
%regresando el error de validaci�n.

a0 = P';

a1 = logsig (W1 * a0 + b1);

a2 = purelin(W2 * a1 + b2);

a = a2;

e = T - a;

valerror = sum(e, 'all') / length(T);