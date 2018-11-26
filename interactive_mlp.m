%Perceptron multicapa

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
configuration.input_file = uigetfile('*.txt','Seleccione el archivo de entradas del dataset');
configuration.output_file = uigetfile('*.txt','Seleccione el archivo de salidas del dataset');

%% Condiciones de finalizacion
prompt = {'Número máximo de épocas','Múltiplo de épocas de validación','Valor máximo de error de época de entrenamiento','Número máximo de incrementos consecutivos de error de valicación'};
title = 'Condiciones de finalización';
dims = [1 1 1 1];
definput = {'1','10','0.1','10'};
answer = inputdlg(prompt,title,dims,definput);
[epochmax, epochval, max_epoch_error_train, numval] = answer{:};
configuration.epochmax = str2double(epochmax);
configuration.epochval = str2double(epochval);
configuration.numval = str2double(numval);
configuration.max_epoch_error_train = str2double(max_epoch_error_train);

%% Distribucion del dataset
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

%% Inicializacion de la arquitectura
parameters = mlp_init(configuration.arch1);

%% Entrenamiento
for epoch = 1:configuration.epochmax
    if mod(epoch,configuration.epochval) == 0
        epoch_validation_error = epoch_validation( configuration.arch2, dataset.valid, parameters );
    else 
        epoch_error = epoch_training( configuration, dataset.train, parameters );
    end
end




