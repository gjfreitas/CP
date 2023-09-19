clc
clear all
close all

% Este script MATLAB permite importar o 
% ficheiro bin?rio criado pelos programas em C
% e produzir um plot. Pode ser muito ?til para
% ver se as condi??es fronteira est?o corretas,
% para ver se n?o h? erros nas fronteiras dos
% subdom?nios, etc.

nx=100;
ny=nx;

fileID = fopen('results_a.bin');
array_MPI = fread(fileID, [ny nx],'double');
fclose(fileID);

L=1;
x=linspace(-L,L,nx);
y=linspace(-L, L,ny);


% Como para as figuras do MATLAB,
% o avan?o numa linha ? um aumento de x 
% (o MATLAB ? em "column-major"),
% para a figura ficar consistente com o 
% programa em C (o nosso programa C escreve 
% em "row-major"), tem que se transpor a matriz.

% array_MPI = rot90(array_MPI');

figure
mesh(x,y,array_MPI')
xlim([-L L])
ylim([-L L])
xlabel('\it{x}')
ylabel('\it{y}')
title('array\_MPI')

load Jacobi_a.mat

% MSE = getMSE(Vnew,array_MPI);

fprintf("MSE: %d\n",getMSE(Vnew,array_MPI));
fprintf("Erro(%%): %.2f\n", round(calculateRelativeError(Vnew, array_MPI), 4));

figure
mesh(x,y,Vnew)
xlim([-L L])
ylim([-L L])
xlabel('\it{x}')
ylabel('\it{y}')
title('Teórico')

function MSE = getMSE(Vnew,array_MPI)

    N2 = 100*100;
    
    matNorm = Vnew;
    MPINorm = array_MPI;
    
    MSE = 1/N2 * sum((MPINorm-matNorm).^2,'all');
end


function relative_error = calculateRelativeError(matrix1, matrix2)
    % Calculate the absolute error between the two matrices
    absolute_error = norm(matrix1 - matrix2, 'fro');
    
    % Calculate the relative error
    relative_error = absolute_error / norm(matrix1, 'fro') * 100;
end



