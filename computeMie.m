addpath(genpath('.'))

simulation = celes_simulation;
particles = celes_particles2;
input = celes_input;
tables = celes_tables;
initialField = celes_initialField;
numerics = celes_numerics;

lmax = 5;
numParts = 700;

radius = linspace(30,1550/2,numParts)';
position = zeros(numParts,3);
refractiveIndex = ones(numParts,1)*1.5;
parameters = zeros(numParts,2);
parameters(:,1) = radius;
parameters(:,2) = refractiveIndex;

particles.positionArray = position;
particles.parameterArray = parameters;

% polar angle of incoming beam/wave, in radians (for Gaussian beams, 
% only 0 and pi are currently possible)
initialField.polarAngle = 0;

% azimuthal angle of incoming beam/wave, in radians
initialField.azimuthalAngle = 0;

% polarization of incoming beam/wave ('TE' or 'TM')
initialField.polarization = 'TM';

% width of beam waist (use 0 or inf for plane wave)
initialField.beamWidth = 0;

% vacuum wavelength (same unit as particle positions and radius)
input.wavelength = 1550;
input.mediumRefractiveIndex = 1;

numerics.lmax = lmax;

input.initialField = initialField;
simulation.numerics = numerics;
simulation.tables = celes_tables;
input.particles = particles;
simulation.input = input;
simulation.tables.pmax = simulation.input.particles.number;
simulation.tables.nmax = simulation.numerics.nmax;

nmax = simulation.tables.nmax;
singleMie = tic;
simulation = simulation.computeMieCoefficients;
timeSingle = toc(singleMie);
simulation = simulation.computeGradMieCoefficients;
% parMie = tic;
% simulation = simulation.computeParallelMieCoefficients;
% timePar = toc(parMie);

mieCoeff = simulation.tables.mieCoefficients;
gradMieCoeff = simulation.tables.gradMieCoefficients;
% figure
% subplot(2,1,1)
% imagesc(radius,1:100,real(mieCoeff))
% title('real mieCoeff');
% colorbar
% subplot(2,1,2)
% imagesc(radius,1:100,imag(mieCoeff))
% title('imag mieCoeff');
% colorbar

% figure
% subplot(2,1,1)
% imagesc(1:nmax,radius,abs(mieCoeff));
% title('abs mieCoeff');
% colorbar
% subplot(2,1,2)
% imagesc(1:nmax,radius,angle(mieCoeff));
% title('phase mieCoeff');
% colorbar

figure
imagesc(1:nmax,radius,abs(mieCoeff));
%title('abs mieCoeff');
xlabel('Expansion order (l)','fontsize',18);
ylabel('Particle Radius (nm)','fontsize',18);

colorbar
