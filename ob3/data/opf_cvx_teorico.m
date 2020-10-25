%cd C:\personal\cvx
clear all
close all
R=30
d3=20
d2=180

cvx_setup


%planteo del problema dentro del entorno cvx
cvx_begin 

cvx_precision high
variables p(3);
variables g(2)

dual variables lambda1
dual variables lambda2
dual variables lambda3
dual variables lambda4
dual variables mu1
dual variables mu2

minimize([1 4]*g)
subject to

   lambda1: [1 0 1]*p+[-1 0]*g==0 %balance en barra 1
   lambda2: [0 1 1]*p+[0 1]*g==d2 %balance en barra 2
   lambda3: d3==[1 -1 0]*p        %balance en barra 3   
   lambda4: [-1 -1 1]*p==0        %flujo DCOPF    
   mu1: [0 1 0]*p<= R             %restriccion linea 2
   mu2: [0 -1 0]*p<= R            %restriccion linea 2  
   g>=0
    
cvx_end


% 
% %parte 2 reducido
% 
% cvx_begin 
% 
% cvx_precision high
% variables p(2)
% variables g(2)
% 
% dual variables lambda1
% dual variables lambda2
% dual variables lambda3
% 
% dual variables mu1
% dual variables mu2
% dual variables mu3
% minimize([1 4]*g)
% subject to
% 
%    lambda1: d3==[1 -1]*p
%    mu1: [0 1]*p<= R
%    lambda2: d2==[1 2]*p+[0 1]*g
%    lambda3: 0==[2 1]*p+[-1 0]*g
%    mu2: [0 1]*g>=0
%    mu3: [1 0]*g>=0
% cvx_end
% 
% 
