function [input,output] = dataset_generate(i,samples)
%Genera un dataset simple de una funcion senoidal y guarda archivos
%'dataset_input.txt' y 'dataset_output.txt'.

input = (-2:1/(samples/4):2)';
output = 1 + sin((pi*i*input)/4);
save('dataset_input.txt','input','-ascii');
save('dataset_output.txt','output','-ascii');
