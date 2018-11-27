%Perceptrón multicapa

%% Configuración de perceptrón
prompt = {'Vector 1 de la arquitectura (capas)','Vector 2 de la arquitectura (funciones de activación)','Factor de aprendizaje (alpha)','Rango de la señal'};
title = 'Parámetros de la red';
dims = [1 1 1 1];
definput = {'[1 2 1]','[2 1]','0.03','[-2 2]'}; 
answer = inputdlg(prompt,title,dims,definput);
[arch1, arch2, alpha, range] = answer{:};
configuration.arch1 = str2num(arch1);
configuration.arch2 = str2num(arch2);
configuration.alpha = str2double(alpha);
configuration.range = str2num(range);
[file, path] = uigetfile('*.txt','Seleccione el archivo de entradas del dataset');
configuration.input_file = fullfile(path, file);
[file, path] = uigetfile('*.txt','Seleccione el archivo de salidas del dataset');
configuration.output_file = fullfile(path, file);

%% Condiciones de finalización
prompt = {'Número máximo de épocas','Múltiplo de épocas de validación','Valor máximo de error de época de entrenamiento','Número máximo de incrementos consecutivos de error de valicación'};
title = 'Condiciones de finalización';
dims = [1 1 1 1];
definput = {'100','10','10','10'};
answer = inputdlg(prompt,title,dims,definput);
[epochmax, epochval, max_epoch_error_train, numval] = answer{:};
configuration.epochmax = str2double(epochmax);
configuration.epochval = str2double(epochval);
configuration.numval   = str2double(numval);
configuration.max_epoch_error_train = str2double(max_epoch_error_train);

%% Distribución del dataset
prompt = {'Porcentaje de datos del conjunto de entrenamiento','Porcentaje de datos del conjunto de validación','Porcentaje de datos del conjunto de prueba'};
title = 'División del conjunto de datos';
dims = [1 1 1];
definput = {'80','10','10'};
answer = inputdlg(prompt,title,dims,definput);
[train_percent, valid_percent, test_percent] = answer{:};
configuration.train_percent = str2double(train_percent);
configuration.valid_percent = str2double(valid_percent);
configuration.test_percent  = str2double(test_percent);
dataset = dataset_divide(configuration.train_percent, configuration.valid_percent, configuration.test_percent, configuration.input_file, configuration.output_file);

%% Workspace cleanup & save configuration
save('configuration.mat','configuration');
clearvars -except configuration dataset
delete historic_*.txt

%% Inicialización de la arquitectura
parameter = mlp_init(configuration.arch1);

%% Entrenamiento
incremento=0;
epoch_validation_error=0;
for epoch = 1:configuration.epochmax
    if mod(epoch,configuration.epochval) == 0
        last_epoch_validation_error = epoch_validation_error;
        epoch_validation_error = epoch_validation( configuration, dataset.valid, parameter );
        if last_epoch_validation_error < epoch_validation_error
            incremento = incremento+1;
        else
            incremento = 0;
        end
        last_epoch_validation_error = epoch_validation_error;
        %Se verifica que no exista sobre entrenamiento y de ser asi el entrenamiento termina.
        if incremento == configuration.numval
            epocas = configuration.epochmax + 1;
            fprintf('\nTermina por early stopping en epoca: %d\n',epoch);
        end
    else 
        [epoch_error, parameter] = epoch_training( configuration, dataset.train, parameter );
    end
%Saving backpropagation calculations for this epoch
    save('parameter.mat','parameter');
end

%% Realizando la etapa de prueba
epoch_test_error = epoch_test( configuration, dataset.test,parameter);

%% Presentación de resultados

disp('Valores finales');
for i = 1:length(parameter)
    fprintf('w%d = ',i);
    fprintf('%f ',parameter(i).w);
    fprintf('\nb%d = ',i);
    fprintf('%f ',parameter(i).b);
    fprintf('\n');
end
fprintf('\nError de epoca:               %f\n',epoch_error);
fprintf('\nError de epoca de validacion: %f\n',epoch_validation_error);
fprintf('\nError de epoca de prueba    : %f\n',epoch_test_error);

historic_weight = importdata("historic_weight.txt");
historic_bias = importdata("historic_bias.txt");

figure;
plot(1:size(historic_weight,1),historic_weight);
xlabel('Weight adjustment');
ylabel('Value');
title('Weight learning');

figure;
plot(1:size(historic_bias,1),historic_bias);
xlabel('Bias adjustment');
ylabel('Value');
title('Bias learning');

figure;
hold on;
test_data = importdata("historic_test.txt");
plot(dataset.test.p,test_data,' x');
plot(dataset.test.p,dataset.test.t,' o');
plot(dataset.train.p,dataset.train.t,'-');
xlabel('Input');
ylabel('Output');
title('MLP');   
legend({'MLP','Test Data'});
hold off;
