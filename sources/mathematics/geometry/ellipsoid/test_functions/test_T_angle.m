[T1 dT1] = compute_T(4,30,30,[800,800,900,0.1],1,1.52,1550);
[T2 dT2] = compute_T(4,30,30,[800,800,900,0.1+pi/2],1,1.52,1550);
[T3 dT3] = compute_T(4,30,30,[800,800,900,pi+0.2],1,1.52,1550);

dT12 = T1-T2;
dT13 = T1-T3;
dT23 = T2-T3;

ddT12 = dT1-dT2;
ddT13 = dT1-dT3;
ddT23 = dT2-dT3;

figure
subplot(4,1,1)
imagesc(abs(dT12))
colorbar
subplot(4,1,2)
imagesc(abs(dT13))
colorbar
subplot(4,1,3)
imagesc(abs(dT23))
colorbar
subplot(4,1,4)
imagesc(abs(T1))
colorbar

figure
subplot(3,1,1)
imagesc(abs(ddT12(:,:,1)))
colorbar
subplot(3,1,2)
imagesc(abs(ddT13(:,:,1)))
colorbar
subplot(3,1,3)
imagesc(abs(ddT23(:,:,1)))
colorbar