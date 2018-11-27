[file, path] = uigetfile('*.txt','Seleccione el archivo de entradas del dataset');
p = load(fullfile(path, file));
[file, path] = uigetfile('*.txt','Seleccione el archivo de entradas del dataset');
t = load(fullfile(path, file));

plot(p,t);