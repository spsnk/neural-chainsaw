function [epoch_error] = epoch_training(arch2, dataset, parameters)
%Esta Funcion ejecuta una época propagando los valores hacia adelante y
%realizando backpropagation, regresando el error de epoca.

%Pre-alocando a
a = zeros(1,length(parameters));

%Procesando cada capa
for layer = 1:length(parameters)
    if mod(layer,2)
        w = parameters(layer).w;
        p = dataset.p;
    else
        w = parameters(layer).w';
        p = dataset.p';
    end
    n = w * p;
    switch arch2(layer)
        case 1
            a(layer) = purelin(n);
        case 2
            a(layer) = logsig(n);
        case 3
            a(layer) = tansig(n);
    end
end

epoch_error = sum(e, 'all') / length(T);