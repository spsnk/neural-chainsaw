function [epoch_error_test] = epoch_test(configuration, dataset, parameter)
epoch_error_test = 0;
for data = 1:length(dataset.p)
   p = dataset.p(data);
   t = dataset.t(data);
%Pre-alocando a
   a = cell(1,length(parameter));
   salidas = zeros(1,length(dataset.p));
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
    salidas(data)=a{length(parameter)};
    e = t - a{length(parameter)};
    epoch_error_test = epoch_error_test + abs(e);
end
epoch_error_test = epoch_error_test / length(dataset.p);

figure;
plot(dataset.p,dataset.t);
plot(dataset.p,salidas);
xlabel('Test');
