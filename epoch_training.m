function [epoch_error] = epoch_training(arch2, dataset, parameters)
%Esta Funcion ejecuta una época propagando los valores hacia adelante y
%realizando backpropagation, regresando el error de epoca.

%Pre-alocando a
a = cell(1,length(parameters));

%% Feed Fordward
for layer = 1:length(parameters)
    if layer == 1
        n = parameters(layer).w * dataset.p';
    else
        n = parameters(layer).w * a{layer-1};
    end
    switch arch2(layer)
        case 1
            a(layer) = {purelin(n)};
        case 2
            a(layer) = {logsig(n)};
        case 3
            a(layer) = {tansig(n)};
    end
end

final_dimension = [length(dataset.t) 1];

e = reshape(dataset.t, final_dimension) - reshape ( a{length(parameters)}, final_dimension);

epoch_error = sum(e)/length(e);

%% Backpropagation

