function [input,output] = dataset_generate(i,samples)

input = (-2:1/(samples/4):2)';
output = 1 + sin((pi*i*input)/4);
save('dataset_input.txt','input','-ascii');
save('dataset_output.txt','output','-ascii');
