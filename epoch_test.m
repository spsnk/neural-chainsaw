function [error] = epoch_test(configuration, dataset, parameter)
%Esta funcion ejecuta la validacion propagando los valores hacia adelante y
%regresando el error de validación.

%% Inicializando
error = 0;
historic_test = fopen('historic_test.txt', 'a+');

for data = 1:length(dataset.p)
   p = dataset.p(data);
   t = dataset.t(data);
%Pre-alocando a
   a = cell(1,length(parameter));

%% Feed Fordward
    for layer = 1:length(parameter)
        if layer == 1
            n = parameter(layer).w * p + parameter(layer).b;
        else
            n = parameter(layer).w * a{layer-1} + parameter(layer).b;
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
    error = error + e;
    fprintf(historic_test, '%f ', a{length(parameter)});
    fprintf(historic_test, '\n');
end
fclose(historic_test);

%% Calculating epoch error
error = error / length(dataset.p);

