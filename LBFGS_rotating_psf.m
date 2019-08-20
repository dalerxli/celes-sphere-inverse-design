addpath(genpath('.'))

%initialize global simulation and parts of it
global LBFGS_simulation;
LBFGS_simulation = celes_simulation;
particles = celes_particles2;
initialField = celes_initialField;
input = celes_input;
numerics = celes_numerics_v2;
solver = celes_solver;
inverseSovler = celes_solver;
output = celes_output;
preconditioner = celes_preconditioner_v2;
inversePreconditioner = celes_preconditioner_v2;

%settings for CUDA code
lmax = 4;
cuda_compile(lmax);
cuda_compile_T(lmax);

%wavelength of interest
lambda = 1550;
input.wavelength = lambda;

%initialFields
initialField.polarAngle = 0;
initialField.azimuthalAngle = 0;


% xl = linspace(-5000,5000,20);
% yl = xl.';
% [xp, yp] = meshgrid(xl,yl);
% zp = zeros(20)+20000;
% r = sqrt(xp.^2+yp.^2);
% r = r./max(max(r));
% image = (r.^2.*exp(-5*r.^2));
% image = image/max(max(image));
% imagePower = sum(sum(image));

%TM = x, TE = y
initialField.polarization = 'TM';
%0 or inf for plane wave
initialField.beamWidth = 0;
initialField.focalPoint = [0,0,0];

%refractive index of scatterer
refractiveIndex = 1.52;
mediumIndex = 1;

%grid of spheres
deviceRadius = 75000;
periodicity = 2440;
xpos1 = 0:periodicity:deviceRadius;
xpos2 = -periodicity:-periodicity:-deviceRadius;
xpos = cat(2,xpos2,xpos1);
ypos = xpos';
zpos = linspace(0,6120,3);
[xx,yy,zz] = meshgrid(xpos,ypos,zpos);

positions = zeros(length(xx(:)),3);
positions(:,1) = [xx(:)];
positions(:,2) = [yy(:)];
positions(:,3) = [zz(:)];
%%
%sphere initial condition
radii = 400*ones(length(xx(:)),1);
radiusArray = radii+200;
parameters = zeros(length(xx(:)),2);
parameters(:,1) = radii;
parameters(:,2) = parameters(:,2)+refractiveIndex;

%upper and lower bounds
max_rad = 1200*ones(length(radii(:)),1);
min_rad = 250*ones(length(radii(:)),1);

%particle properties
particles.type = 'sphere';
particles.parameterArray = parameters;
particles.positionArray = positions;
input.mediumRefractiveIndex = mediumIndex;

%solver numerics
numerics.lmax = lmax;
numerics.particleDistanceResolution = 0.2;
numerics.gpuFlag = true;
numerics.polarAnglesArray = 0:pi/1e3:pi;
numerics.azimuthalAnglesArray = 0:pi/1e3:2*pi;

%solver properties
solver.type = 'BiCGStab';
solver.tolerance = 1e-3;
solver.maxIter = 1000;
solver.restart = 1000;
inverseSolver = solver;

%preconditioner properties
preconditioner.type = 'blockdiagonal';
numerics.partitionEdgeSizes = [15000,15000,2040];  
inversePreconditioner.type = 'blockdiagonal';
solver.preconditioner = preconditioner;
inverseSolver.preconditioner = inversePreconditioner;

%fom
global LBFGS_points;
global LBFGS_image;
%%
R = 10000;
z = 100000:12500:300000;
t = linspace(pi/4,2*pi,length(z));

x = R*t.*cos(3*t);
y = R*t.*sin(3*t);

plot3(x,y,z);
points(1,1) = x(1);
points(1,2) = y(1);
points(1,3) = z(1);
image(1) = 100;

points(2,1) = x(2);
points(2,2) = y(2);
points(2,3) = z(1);
image(2) = 0;

for i = 2:length(z)-1
    image = [image, 100, 0, 0];

    pt1 = [x(i),y(i), z(i)];
    pt2 = [x(i-1),y(i-1),z(i)];
    pt3 = [x(i+1),y(i+1),z(i)];
    
    points = [points; pt1; pt2; pt3];
end

%%
LBFGS_points = points;
LBFGS_image = image;

%put into simulation object;
input.initialField = initialField;
input.particles = particles;
LBFGS_simulation.input = input;
LBFGS_simulation.tables.pmax = LBFGS_simulation.input.particles.number;
numerics.solver = solver;
numerics.inverseSolver = inverseSolver;
LBFGS_simulation.numerics = numerics;
LBFGS_simulation.tables = celes_tables;
LBFGS_simulation.output = output;
LBFGS_simulation.tables.nmax = LBFGS_simulation.numerics.nmax;
LBFGS_simulation.tables.pmax = LBFGS_simulation.input.particles.number;
LBFGS_simulation = LBFGS_simulation.computeInitialFieldPower;
LBFGS_simulation = LBFGS_simulation.computeTranslationTable;
LBFGS_simulation.input.particles = LBFGS_simulation.input.particles.compute_maximal_particle_distance;

if strcmp(LBFGS_simulation.numerics.solver.preconditioner.type,'blockdiagonal')
    fprintf(1,'make particle partition ...');
    partitioning = make_particle_partion(LBFGS_simulation.input.particles.positionArray,LBFGS_simulation.numerics.partitionEdgeSizes);
    LBFGS_simulation.numerics.partitioning = partitioning;
    LBFGS_simulation.numerics.solver.preconditioner.partitioning = partitioning;
    LBFGS_simulation.numerics.inverseSolver.preconditioner.partitioning = partitioning;
    fprintf(1,' done\n');
    LBFGS_simulation = LBFGS_simulation.numerics.prepareW(LBFGS_simulation);
    LBFGS_simulation.numerics.solver.preconditioner.partitioningIdcs = LBFGS_simulation.numerics.partitioningIdcs;
    LBFGS_simulation.numerics.inverseSolver.preconditioner.partitioningIdcs = LBFGS_simulation.numerics.partitioningIdcs;
end
%%

celes_func = @(x) CELES_LBFGS_iteration(x);

opts = struct('x0',radii(:),'m',10,'maxIts',50,'maxTotalIts',250);

tic
[r_f, fom_f, info] = lbfgsb(celes_func,min_rad,max_rad,opts);
lbfgs_time = toc;

%%

size_x = 500;
size_y = 500;
xi = linspace(-deviceRadius,deviceRadius,size_x);
yi = linspace(-deviceRadius,deviceRadius,size_x);
zi = z;

[x_i, y_i,z_i] = meshgrid(xi,yi,zi);
field_pts(:,1) = x_i(:);
field_pts(:,2) = y_i(:);
field_pts(:,3) = z_i(:);

LBFGS_simulation.output.fieldPoints = field_pts;
E_scat = compute_scattered_field_opt(LBFGS_simulation,field_pts);
E_in = compute_initial_field(LBFGS_simulation);

E_tot=E_scat+E_in;

%%
I = reshape(gather(sum(abs(E_tot).^2,2)),[500,500,17]);
%%
figure()
subplot(2,2,1)
imagesc(I(:,:,1))
colorbar

subplot(2,2,2)
imagesc(I(:,:,5))
colorbar



