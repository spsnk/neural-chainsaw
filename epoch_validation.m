function [validation_error] = epoch_validation(arch2, dataset, parameter)
%Esta funcion ejecuta la validacion propagando los valores hacia adelante y
%regresando el error de validación.

%Pre-alocando a
a = cell(1,length(parameter));

%Procesando cada capa
for layer = 1:length(parameter)
    if layer == 1
        n = parameter(layer).w * dataset.p';
    else
        n = parameter(layer).w * a{layer-1};
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

e = reshape(dataset.t, final_dimension) - reshape ( a{length(parameter)}, final_dimension);

validation_error = sum(e)/length(e);

