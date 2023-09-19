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

h = x(2)-x(1);

N = nx;
Vold = zeros(N,N);

% Condições de fronteira
Vold(1,:) = (1 + x)./4;
Vold(:,1) = (1 + y)./4;
Vold(end,:) = (3 + x)./4;
Vold(:,end) = (3 + y)./4;

% Jacobi
for k = 1:maxit
    Vnew = Vold;
    for i = 2:N-1
        for j = 2:N-1
            f = 7*sin(2*pi*x(i))*cos(3*pi*x(i))*sin(2*pi*y(j))*cos(3*pi*y(j));  
            Vnew(i,j)= 0.25*(Vold(i+1,j)+Vold(i-1,j)+Vold(i,j+1)+Vold(i,j-1)-h^2*f);
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

save('Jacobi_a', 'Vnew')
