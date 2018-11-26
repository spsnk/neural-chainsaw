%Perceptr�n multicapa

%% Configuraci�n de perceptr�n
prompt = {'Vector 1 de la arquitectura (capas)','Vector 2 de la arquitectura (funciones de activaci�n)','Factor de aprendizaje (alpha)','Rango de la se�al'};
title = 'Par�metros de la red';
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

%% Condiciones de finalizaci�n
prompt = {'N�mero m�ximo de �pocas','M�ltiplo de �pocas de validaci�n','Valor m�ximo de error de �poca de entrenamiento','N�mero m�ximo de incrementos consecutivos de error de valicaci�n'};
title = 'Condiciones de finalizaci�n';
dims = [1 1 1 1];
definput = {'100','10','0.1','10'};
answer = inputdlg(prompt,title,dims,definput);
[epochmax, epochval, max_epoch_error_train, numval] = answer{:};
configuration.epochmax = str2double(epochmax);
configuration.epochval = str2double(epochval);
configuration.numval = str2double(numval);
configuration.max_epoch_error_train = str2double(max_epoch_error_train);

%% Distribuci�n del dataset
prompt = {'Porcentaje de datos del conjunto de entrenamiento','Porcentaje de datos del conjunto de validaci�n','Porcentaje de datos del conjunto de prueba'};
title = 'Divisi�n del conjunto de datos';
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

%% Inicializaci�n de la arquitectura
parameter = mlp_init(configuration.arch1);

%% Entrenamiento
for epoch = 1:configuration.epochmax
    if mod(epoch,configuration.epochval) == 0
        epoch_validation_error = epoch_validation( configuration, dataset.valid, parameter );
    else 
        [epoch_error, parameter] = epoch_training( configuration, dataset.train, parameter );
    end
%Saving backpropagation calculations for this epoch
    save('parameter.mat','parameter');
end

%% Presentaci�n de resultados



