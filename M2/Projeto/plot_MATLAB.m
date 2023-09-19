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

fileID = fopen('results_c.bin');
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

figure
mesh(x,y,array_MPI')
xlim([-L L])
ylim([-L L])
xlabel('\it{x}')
ylabel('\it{y}')
% title('array\_MPI')
