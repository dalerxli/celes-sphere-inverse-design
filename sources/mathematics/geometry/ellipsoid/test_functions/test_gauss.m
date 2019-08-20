close all
clear all

Nx = 10;
Ny = 10;

[x,wx] = generate_gauss_weights_abscissae(Nx,0,2*pi);
[y,wy] = generate_gauss_weights_abscissae(Ny,0,2*pi);

[wxx,wyy] = meshgrid(wx,wy);
weightmap = wxx.*wyy;
[xx,yy] = meshgrid(x,y);

total = sum(sum(sin(xx).*sin(yy).*weightmap));