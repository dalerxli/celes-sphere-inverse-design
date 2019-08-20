%ellip geometry
a = 300;
b = 400;
c = 350;
%wavelength
lambda = 600;
%refractive indices
ni = 1;
ns = 1.52;
%max expansion order
lmax = 4;
%numerical derivative
diff = 1;

[J11,J12,J21,J22,dJ11,dJ12,dJ21,dJ22] = compute_J_ellip_celes_posm(lmax,40,40,a,b,c,ni,ns,lambda,3);
[aJ11,aJ12,aJ21,aJ22,~,~,~,~] = compute_J_ellip_celes_posm(lmax,40,40,a+diff,b,c,ni,ns,lambda,3);
[bJ11,bJ12,bJ21,bJ22,~,~,~,~] = compute_J_ellip_celes_posm(lmax,40,40,a,b+diff,c,ni,ns,lambda,3);
[cJ11,cJ12,cJ21,cJ22,~,~,~,~] = compute_J_ellip_celes_posm(lmax,40,40,a,b,c+diff,ni,ns,lambda,3);
[T dT] = compute_T(lmax,40,40,[a,b,c],ni,ns,lambda);
[T2 ~] = compute_T(lmax,40,40,[a+1,b,c],ni,ns,lambda);

figure
imagesc(abs(T))
colorbar
figure
imagesc(abs(dT(:,:,1)))
colorbar
figure
imagesc(abs(T-T2))
colorbar

daJ11 = aJ11-J11;
daJ12 = aJ12-J12;
daJ21 = aJ21-J21;
daJ22 = aJ22-J22;
dbJ11 = bJ11-J11;
dbJ12 = bJ12-J12;
dbJ21 = bJ21-J21;
dbJ22 = bJ22-J22;
dcJ11 = cJ11-J11;
dcJ12 = cJ12-J12;
dcJ21 = cJ21-J21;
dcJ22 = cJ22-J22;

ddaJ11 = daJ11-dJ11(:,:,1)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddaJ11))
colorbar
title('J11a');
subplot(3,1,2)
imagesc(abs(dJ11(:,:,1)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(daJ11))
colorbar

ddbJ11 = dbJ11-dJ11(:,:,2)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddbJ11))
title('J11b');
colorbar
subplot(3,1,2)
imagesc(abs(dJ11(:,:,2)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dbJ11))
colorbar

ddcJ11 = dcJ11-dJ11(:,:,3)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddcJ11))
colorbar
title('J11c')
subplot(3,1,2)
imagesc(abs(dJ11(:,:,3)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dcJ11))
colorbar

ddaJ12 = daJ12-dJ12(:,:,1)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddaJ12))
colorbar
title('J12a');
subplot(3,1,2)
imagesc(abs(dJ12(:,:,1)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(daJ12))
colorbar

ddbJ12 = dbJ12-dJ12(:,:,2)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddbJ12))
title('J12b');
colorbar
subplot(3,1,2)
imagesc(abs(dJ12(:,:,2)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dbJ12))
colorbar

ddcJ12 = dcJ12-dJ12(:,:,3)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddcJ12))
colorbar
title('J12c')
subplot(3,1,2)
imagesc(abs(dJ12(:,:,3)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dcJ12))
colorbar

ddaJ21 = daJ21-dJ21(:,:,1)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddaJ21))
title('J21a')
colorbar
subplot(3,1,2)
imagesc(abs(dJ21(:,:,1)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(daJ21))
colorbar

ddbJ21 = dbJ21-dJ21(:,:,2)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddbJ21))
title('J21b')
colorbar
subplot(3,1,2)
imagesc(abs(dJ21(:,:,2)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dbJ21))
colorbar

ddcJ21 = dcJ21-dJ21(:,:,3)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddcJ21))
title('J21c')
colorbar
subplot(3,1,2)
imagesc(abs(dJ21(:,:,3)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dcJ21))
colorbar

ddaJ22 = daJ22-dJ22(:,:,1)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddaJ22))
title('J22a')
colorbar
subplot(3,1,2)
imagesc(abs(dJ22(:,:,1)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(daJ22))
colorbar

ddbJ22 = dbJ22-dJ22(:,:,2)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddbJ22))
title('J22b')
colorbar
subplot(3,1,2)
imagesc(abs(dJ22(:,:,2)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dbJ22))
colorbar

ddcJ22 = dcJ22-dJ22(:,:,3)*diff;
figure
subplot(3,1,1)
imagesc(abs(ddcJ22))
title('J22c')
colorbar
subplot(3,1,2)
imagesc(abs(dJ22(:,:,3)*diff));
colorbar
subplot(3,1,3)
imagesc(abs(dcJ22))
colorbar