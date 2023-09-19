close all
clear all
clc
tic
maxit = 5e5;
tolerancia = 1e-6;

nx = 100;
ny = nx;
L = 1;
x = linspace(-L,L,nx);
y = linspace(-L,L,ny);


% h = 2/nx;
% h = 2/(nx-1);
h = x(2)-x(1); 

N = nx;
Vold=zeros(N,N);

% Jacobi
for k = 1:maxit
    Vnew = Vold;
    for i = 1:N
        for j = 1:N
            f = 7*sin(2*pi*x(i))*cos(3*pi*x(i))*sin(2*pi*y(j))*cos(3*pi*y(j));

            [i0,i1,j0,j1] = getIJ(i,j,N);

             Vnew(i,j)= 0.25*(Vold(i0,j)+Vold(i1,j)+Vold(i,j0)+Vold(i,j1)-h^2*f);
        end
    end
    diff = sqrt(sum(sum((Vnew - Vold).^2))) / sqrt(sum(sum(Vnew.^2)));
    if diff < tolerancia
        break;
    end
	Vold=Vnew;
end 
toc
figure;
mesh(x,y,Vnew)
xlim([-L L])
ylim([-L L])
xlabel('\it{x}')
ylabel('\it{y}')

save('Jacobi_b', 'Vnew')

function [i0,i1,j0,j1] = getIJ(i,j,N)
    j1= j + 1;
    j0 = j - 1;

    if (j0 <= 0)
        j0 = j0+N;
    elseif (j1 > N)
        j1 = j1-N;
    end

    i1= i + 1;
    i0 = i - 1;

    if (i0 <= 0)
        i0 = i0+N;
    elseif (i1 > N)
        i1 = i1-N;
    end
end