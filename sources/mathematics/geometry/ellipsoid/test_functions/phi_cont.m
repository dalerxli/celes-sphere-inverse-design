clear all;
close all;
clc;

phi_ang = 0:5e-2:3*pi;
T_matrices = zeros(30,30,length(phi_ang));

for i = 1:length(phi_ang)
    [Temp ~] = compute_T(3,30,30,[750,800,700,phi_ang(i)],1,1.52,1550);
    T_matrices(:,:,i) = Temp;
end