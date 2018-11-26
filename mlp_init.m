function [parameters] = mlp_init (arquitecture)
%Inicializa los pesos y bias basado en la arquitectura dada como argumento
% por ejemplo: v1 = [1 2 1];
%Regresa y guarda los valores en el archivo 'parameters.mat'

parameters = struct ( 'w', cell([1 2]), 'b', cell([1 2]));

for i=2:length(arquitecture)
   parameters(i-1).w = -1 + 2.*rand([1 (arquitecture(i) * arquitecture(i-1))]);
   parameters(i-1).b = -1 + 2.*rand([1 arquitecture(i)]);
end

save('parameters.mat','parameters');