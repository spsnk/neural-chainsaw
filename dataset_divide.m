function [dataset] = dataset_divide ( train_percent, valid_percent, test_percent, dataset_input, dataset_output )
%Divide un dataset (en texto plano) dado en subconjuntos de entrenamiento 
%validacion y prueba. Regresa y guarda en archivo 'dataset.mat' la
%estructura resultante. Ejemplo:
%   dataset.train.p -> input del conjunto de entrenamiento
%   dataset.valid.t -> output del conjunto de validacion
%   dataset.test    -> conjunto de prueba, contiene p y t

p = load(dataset_input);
t = load(dataset_output);

dataset_size = length(p);

dataset.full.p = p;
dataset.full.t = t;

valid_size = floor ( dataset_size * valid_percent/100 );
test_size = floor ( dataset_size * test_percent/100 );

dataset.valid.p = p(1:ceil(dataset_size/valid_size):dataset_size);
p(1:ceil(dataset_size/valid_size):dataset_size) = [];
dataset.valid.t = t(1:ceil(dataset_size/valid_size):dataset_size);
t(1:ceil(dataset_size/valid_size):dataset_size) = [];

dataset.test.p = p(1:ceil(length(p)/test_size):length(p));
p(1:ceil(length(p)/test_size):length(p)) = [];
dataset.test.t = t(1:ceil(length(t)/test_size):length(t));
t(1:ceil(length(t)/test_size):length(t)) = [];

dataset.train.p = p;
dataset.train.t = t;

save('dataset.mat','dataset');