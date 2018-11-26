function [epoch_error] = epoch_training(configuration, dataset, parameters)
%Esta Funcion ejecuta una época propagando los valores hacia adelante y
%realizando backpropagation, regresando el error de epoca.

%% Inicializando
epoch_error = 0;
for data = 1:length(dataset.p)
   p = dataset.p(data);
   t = dataset.t(data);
%Pre-alocando a
   a = cell(1,length(parameters));

%% Feed Fordward
    for layer = 1:length(parameters)
        if layer == 1
            n = parameters(layer).w * p;
        else
            n = parameters(layer).w * a{layer-1};
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
    %final_dimension = [length(dataset.t) 1];
    %e = reshape(t, final_dimension) - reshape (a{length(parameters)}, final_dimension);
    e = t - a{length(parameters)};
    epoch_error = epoch_error + e;

%% Backpropagation
    s = cell(1,length(parameters));
    for layer = length(parameters):-1:1
%Calculo de sensitividades
        switch configuration.arch2(layer)
            case 1
                f = 1;
            case 2
                dim = length(a{layer});
                f = zeros(dim);
                for a_fill = 1:dim
                    a_vect(dim) = (1-a{layer}(a_fill))*a{layer}(a_fill);
                end
                f(1:dim+1:dim^2) = a_vect;
            case 3
                dim = length(a{layer});
                f = zeros(dim);
                for a_fill = 1:dim
                    a_vect(dim) = 1 - (a{layer}(a_fill))^2;
                end
                f(1:dim+1:dim^2) = a_vect;
        end
        if layer == length(parameters)
            s{layer} = -2*f*e;
        else 
            s{layer} = f * parameters(layer).w * s{layer+1};
        end
%Aplicación de reglas de aprendizaje
        if layer == 1
            weight_adjust = parameters(layer).w - configuration.alpha * s{layer} .* p;
        else
            weight_adjust = parameters(layer).w - configuration.alpha * s{layer} .* a{layer-1}';
        end
        parameters(layer).w = weight_adjust;
    end
end