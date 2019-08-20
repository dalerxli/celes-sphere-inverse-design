lmax = 4;
Ntheta = 20;
Nphi = 20;
a = 200;
b = 100;
c = 50;
ni = 1;
ns = 2;
lambda = 100;
nu = 3;

[J11,J12,J21,J22] = compute_J_ellip2(lmax,Ntheta,Nphi,a,b,c,ni,ns,lambda,nu);
[T dT] = compute_T(lmax,Ntheta,Nphi,[a,b,c],ni,ns,lambda);
figure
imagesc(abs(T)
figure
imagesc(abs(J11))
colorbar
figure
imagesc(abs(J12))
colorbar
figure
imagesc(abs(J21))
colorbar
figure
imagesc(abs(J22))
colorbar
nu = 1;

[J11,J12,J21,J22] = compute_J_ellip2(lmax,Ntheta,Nphi,a,b,c,ni,ns,lambda,nu);

figure
imagesc(abs(J11))
colorbar
figure
imagesc(abs(J12))
colorbar
figure
imagesc(abs(J21))
colorbar
figure
imagesc(abs(J22))
colorbar