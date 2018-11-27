function [parameter] = mlp_init (arquitecture)
%Inicializa los pesos y bias basado en la arquitectura dada como argumento
% por ejemplo: v1 = [1 2 1];
%Regresa y guarda los valores en el archivo 'parameters.mat'

parameter = struct ( 'w', cell([1 2]), 'b', cell([1 2]));

for i=2:length(arquitecture)
   if i == 2
       dimsw = [arquitecture(i) 1];
   else
       dimsw = [arquitecture(i)  arquitecture(i-1)];
   end
   dimsb = [arquitecture(i) 1];
   parameter(i-1).w = -1 + 2.*rand(dimsw);
   parameter(i-1).b = -1 + 2.*rand(dimsb);
end

save('parameter.mat','parameter');