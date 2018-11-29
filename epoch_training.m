function [epoch_error, parameter] = epoch_training(configuration, dataset, parameter)
%Esta Funcion ejecuta una época propagando los valores hacia adelante y
%realizando backpropagation, regresando el error de epoca.

%% Inicializando
epoch_error = 0;

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
    epoch_error = epoch_error + abs(e);

%% Backpropagation
    s = cell(1,length(parameter));
    f = cell(1,length(parameter));
    for layer = length(parameter):-1:1
%Calculo de sensitividades
        a_vect = [];
        switch configuration.arch2(layer)
            case 1
                f{layer} = 1;
            case 2
                dim = length(a{layer});
                f{layer} = zeros(dim);
                for a_fill = 1:dim
                    a_vect(a_fill) = ( 1 - a{layer}(a_fill) ) * a{layer}(a_fill);
                end
                f{layer}(1:dim+1:dim^2) = a_vect;
            case 3
                dim = length(a{layer});
                f{layer} = zeros(dim);
                for a_fill = 1:dim
                    a_vect(a_fill) = 1 - (a{layer}(a_fill))^2;
                end
                f{layer}(1:dim+1:dim^2) = a_vect;
        end
        if layer == length(parameter)
            s{layer} = -2*f{layer}*e;
        else 
            s{layer} = f{layer} * parameter(layer+1).w' * s{layer+1};
        end
%Aplicación de reglas de aprendizaje
        if layer == 1
            weight_adjust = parameter(layer).w - configuration.alpha * s{layer} * p;
        else
            weight_adjust = parameter(layer).w - configuration.alpha * s{layer} * a{layer-1}';
        end
        bias_adjust = parameter(layer).b - configuration.alpha * s{layer};
        parameter(layer).w = weight_adjust;
        parameter(layer).b = bias_adjust;
%Saving historical data
    end
end

%% Calculating epoch error
epoch_error = epoch_error / length(dataset.p);