function [valdation_epoch_error] = epoch_validation(configuration, dataset, parameter)
%Esta funcion ejecuta la validacion propagando los valores hacia adelante y
%regresando el error de validación.

%% Inicializando
valdation_epoch_error = 0;
for data = 1:length(dataset.p)
   p = dataset.p(data);
   t = dataset.t(data);
%Pre-alocando a
   a = cell(1,length(parameter));

%% Feed Fordward
    for layer = 1:length(parameter)
        if layer == 1
            n = parameter(layer).w * p;
        else
            n = parameter(layer).w * a{layer-1};
        end
        switch configuration.arch2(layer)
            case 1
                a{layer} = purelin(n);
            case 2
                a{layer} = logsig(n);
            case 3
                a{layer} = tansig(n);
        end
    end
    e = t - a{length(parameter)};
    valdation_epoch_error = valdation_epoch_error + abs(e);
end

%% Calculating epoch error
valdation_epoch_error = valdation_epoch_error / length(dataset.p);

