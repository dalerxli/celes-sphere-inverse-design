function [fom, gradient] = CELES_LBFGS_iteration(r_k)
%%declare global properties
%simulation to be updated per iteration
global LBFGS_simulation;
%points to calculate figure of merit
global LBFGS_points;
%image figure of merit
global LBFGS_image;

%input r_k into simulation
LBFGS_simulation.input.particles.parameterArray(:,1) = r_k;

%%forward simulation
%compute mie coeff
LBFGS_simulation = LBFGS_simulation.computeParallelMieCoefficients;

%prepare preconditioner for solver
LBFGS_simulation = LBFGS_simulation.numerics.solver.preconditioner.prepareM(LBFGS_simulation);
LBFGS_simulation = LBFGS_simulation.computeInitialFieldCoefficients;

%solve linear system and clear preconditioner
LBFGS_simulation = LBFGS_simulation.computeScatteredFieldCoefficients();
LBFGS_simulation.numerics.solver.preconditioner.factorizedMasterMatrices = [];

%compute total electric field
%E_i = gather(compute_initial_field_opt(LBFGS_simulation,LBFGS_points));
E_s = gather(compute_scattered_field_opt(LBFGS_simulation,LBFGS_points));
%E_t = E_i + E_s;
E_t = E_s;

%compute figure of merit
fom = sum((LBFGS_image(:)-sum(abs(gather(E_t)).^2,2)).^2);

%%inverse simulation
%compute grad mie coefficients
LBFGS_simulation = LBFGS_simulation.computeParallelGradMieCoefficients;

%coupled coefficients calculation
LBFGS_simulation = LBFGS_simulation.computeCoupledScatteringCoefficients;

%prepare inverse preconditioner
LBFGS_simulation = LBFGS_simulation.numerics.inverseSolver.preconditioner.prepareMt(LBFGS_simulation);

%solve the system
LBFGS_simulation = LBFGS_simulation.computeImageAdjointCoefficients(LBFGS_image,E_t,LBFGS_points);

%clear preconditioner
LBFGS_simulation.numerics.inverseSolver.preconditioner.factorizedMasterMatrices = [];
grad1 = compute_adjoint_grad_intensity(LBFGS_simulation,1);
% grad1n = grad1/sqrt(sum(abs(grad1).^2));

% max_step_size = 1;
% gradient = max_step_size*grad1n;
gradient = grad1;