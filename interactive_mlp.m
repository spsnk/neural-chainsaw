%Perceptron multicapa

%% Parametros
prompt = {'Vector 1 de la arquitectura (capas)','Vector 2 de la arquitectura (funciones de activación)','Factor de aprendizaje (alpha)','Rango de la señal'};
title = 'Parámetros de la red';
dims = [1 1 1 1];
definput = {'[1 2 1]','[2 1]','0.03','[-2 2]'}; 
answer = inputdlg(prompt,title,dims,definput);
[arch1, arch2, alpha, range] = answer{:};
arch1 = str2num(arch1);
arch2 = str2num(arch2);
alpha = str2double(alpha);
range = str2num(range);
input_file = uigetfile('*.txt','Seleccione el archivo de entradas del dataset');
output_file = uigetfile('*.txt','Seleccione el archivo de salidas del dataset');

%% Condiciones de finalizacion
prompt = {'Número máximo de épocas','Múltiplo de épocas de validación','error_epoch_train','Número máximo de incrementos consecutivos de error de valicación'};
title = 'Condiciones de finalización';
dims = [1 1 1 1];
definput = {'1000','10','0.1','10'};
answer = inputdlg(prompt,title,dims,definput);
[epochmax, epochval, error_epoch_train, numval] = answer{:};
epochmax = str2double(epochmax);
epochval = str2double(epochval);
error_epoch_train = str2double(error_epoch_train);
numval = str2double(numval);

%% Distribucion del dataset

