%Perceptrón multicapa
%% Workspace preparation
fclose('all');
reload = false;

%% Usar configuracion previa
answer = questdlg ( '¿Desea cargar configuración previa o introducir datos?', 'Configuración','Cargar','Introducir','Cargar');
switch answer
    case 'Cargar'
        reload = true;
        load('configuration.mat');
        load('parameter.mat');
        load('dataset.mat');
end

%% Configuración de perceptrón
if ~reload
prompt = {'Vector 1 de la arquitectura (capas)','Vector 2 de la arquitectura (funciones de activación)','Factor de aprendizaje (alpha)','Rango de la señal'};
title = 'Parámetros de la red';
dims = [1 1 1 1];
definput = {'[1 2 1]','[2 1]','0.0003','[-2 2]'}; 
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
end

%% Condiciones de finalización
prompt = {'Número máximo de épocas','Múltiplo de épocas de validación','Valor máximo de error de época de entrenamiento','Número máximo de incrementos consecutivos de error de valicación'};
title = 'Condiciones de finalización';
dims = [1 1 1 1];
definput = {'100','10','0.000001','50'};
answer = inputdlg(prompt,title,dims,definput);
[epochmax, epochval, max_epoch_error_train, numval] = answer{:};
configuration.epochmax = str2double(epochmax);
configuration.epochval = str2double(epochval);
configuration.numval   = str2double(numval);
configuration.max_epoch_error_train = str2double(max_epoch_error_train);

%% Distribución del dataset
if ~reload
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
end

%% Inicialización de la arquitectura
if ~reload
parameter = mlp_init(configuration.arch1);
end

%% Workspace cleanup & save configuration
if ~reload
save('configuration.mat','configuration');
delete historic_*.txt
end
configuration.historic_weight = fopen('historic_weight.txt', 'a+');
configuration.historic_bias   = fopen('historic_bias.txt', 'a+');
clearvars -except configuration dataset parameter

%% Entrenamiento
tic
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
            fprintf('\nTermina por early stopping (incremento) en epoca: %d\n',epoch);
            epoch = configuration.epochmax + 1;
        end
    else 
        [epoch_error, parameter] = epoch_training( configuration, dataset.train, parameter );
        if epoch_error < configuration.max_epoch_error_train
            fprintf('\nTermina por early stopping (error de entrenamiento) en epoca: %d\n',epoch);
            epoch = configuration.epochmax + 1;
        end
    end
%Saving backpropagation calculations for this epoch
    save('parameter.mat','parameter');
end
fclose(configuration.historic_weight);
fclose(configuration.historic_bias);
toc
%% Realizando la etapa de prueba
epoch_test_error = epoch_test( configuration, dataset.test, parameter );

%% Presentación de resultados

disp('Configuración');
disp(configuration);
disp('Valores finales');
for i = 1:length(parameter)
    fprintf('w%d = [ ',i);
    fprintf('%f ',parameter(i).w);
    fprintf(']\nb%d = [ ',i);
    fprintf('%f ',parameter(i).b);
    fprintf(']\n');
end
fprintf('\nError de epoca:               %f\n',epoch_error);
fprintf('Error de epoca de validacion: %f\n',epoch_validation_error);
fprintf('Error de epoca de prueba    : %f\n',epoch_test_error);

figure('Name','Multi Layer Perceptron Output');
hold on;
test_data = importdata("historic_test.txt");
plot(dataset.test.p,test_data,'b:x');
plot(dataset.test.p,dataset.test.t,'r o');
plot(dataset.train.p,dataset.train.t,'r-');
xlabel('Input');
ylabel('Output');
title('MLP');
legend({'MLP','Test Data'});
saveas(gcf,'fig_mlp_test','png');
hold off;

figure('Name','Weight evolution');
historic_weight = importdata("historic_weight.txt");
plot(1:size(historic_weight,1),historic_weight);
xlabel('Weight adjustment');
ylabel('Value');
title('Weight learning');
saveas(gcf,'fig_weight_learning','png');

figure('Name','Bias evolution');
historic_bias = importdata("historic_bias.txt");
plot(1:size(historic_bias,1),historic_bias);
xlabel('Bias adjustment');
ylabel('Value');
title('Bias learning');
saveas(gcf,'fig_bias_learning','png');

clearvars historic*

