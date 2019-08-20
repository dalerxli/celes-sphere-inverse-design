%compute J matrices 11,12,21,22 using gaussian quadrature for computation
%   of full T matrix of an ellipsoid
%formulas are taken from 
%'Differential Cross Section of a Dielectric Ellipsoid by the T-Matrix 
%   Extended Boundary Condition Method', J.Schneider, and I. Peden. IEEE 
%   Transactions on Antennas and Propagation. Vol 36, NO 9, September 1988

%lmax: integer specifying maximum order
%Ntheta: number of points for Gaussian quadrature along polar
%Nphi: number of points for Gaussian quadrature along azimuthal
%a,b,c: axes of ellipsoid
%ni,ns: refractive index of medium and particle respectively
%lambda: wavelength of light
%nu: internal (1) or external (3)
%J_switch: string specifying which J to calculate.

function J = compute_J_ellip(lmax,Ntheta,Nphi,a,b,c,ni,ns,lambda,nu,J_switch)
tic
%preallocate memory for J
nmax = jmult_max(1,lmax);
J = zeros(nmax/2);
%setup for gaussian quadrature in two dimensions
[theta_i,wtheta_i] = generate_gauss_weights_abscissae(Ntheta,0,pi/2);
[phi_i,wphi_i] = generate_gauss_weights_abscissae(Nphi,0,pi);
%generate r, theta, phi corresponding to the surface of the ellisoid
[theta_map,phi_map] = meshgrid(theta_i,phi_i);
r = ellip_rad(theta_map,phi_map,a,b,c);
%generate weight map
[wt,wp] = meshgrid(wtheta_i,wphi_i);
weight_map = wt.*wp;
%k vector freespace and in scatterer
k = 2*pi*ni/lambda;
ks = 2*pi*ns/lambda;
%associated legendre polynomials
P_lm = legendre_normalized_angular(theta_map,lmax);
ct = cos(theta_map);
switch J_switch
    case '11'
        for li=1:lmax
            for mi = -li:li
                ni = multi2single_index(1,1,li,mi,lmax);
                for lp=1:lmax
                    for mp=-lp:lp
                        np = multi2single_index(1,1,lp,mp,lmax);
                        %generate prefactor here, can speed up using cases
                        %where it is only zero
                        prefactor = selection_rules(li,mi,lp,mp,1)*gamma(li,-mi)*gamma(lp,mp)*exp(1i*phi_map*(mp-mi));
                        %make sure we use the right signed associated legendre
                        %polynomial
                        if mp < 0
                            plm_p = P_lm{lp+1,abs(mp)+1}*(-1)^mp*factorial(lp-abs(mp))/factorial(lp+abs(mp));
                        else
                            plm_p = P_lm{lp+1,abs(mp)+1};
                        end
                        if mi > 0
                            plm_i = P_lm{li+1,abs(mi)+1}*(-1)^mi*factorial(li-mi)/factorial(li+mi);
                        else
                            plm_i = P_lm{li+1,abs(mi)+1};
                        end
                        angular = 1i*mp.*plm_p.*assoc_legendre_deriv(P_lm,li,-mi,ct)+1i*mi.*plm_i.*assoc_legendre_deriv(P_lm,lp,mp,ct);
                        J(ni,np) = sum(sum(weight_map.*prefactor.*r.^2.*angular.*jh_product(r,k,ks,li,lp,nu,1)));
                    end
                end
            end
        end
                        
    case '12'
        for li=1:lmax
            for mi = -li:li
                ni = multi2single_index(1,1,li,mi,lmax);
                for lp=1:lmax
                    for mp=-lp:lp
                        np = multi2single_index(1,1,lp,mp,lmax);
                        prefactor = selection_rules(li,mi,lp,mp,2)*gamma(li,-mi)*gamma(lp,mp)*exp(1i*phi_map*(mp-mi));
                        %make sure we use the right signed associated legendre
                        %polynomial
                        if mp < 0
                            plm_p = P_lm{lp+1,abs(mp)+1}*(-1)^mp*factorial(lp-abs(mp))/factorial(lp+abs(mp));
                        else
                            plm_p = P_lm{lp+1,abs(mp)+1};
                        end
                        if mi > 0
                            plm_i = P_lm{li+1,abs(mi)+1}*(-1)^mi*factorial(li-mi)/factorial(li+mi);
                        else
                            plm_i = P_lm{li+1,abs(mi)+1};
                        end
                        term1 = -1/k*r.*jh_product(r,k,ks,li,lp,nu,2).*(mi*mp*plm_p.*plm_i./sin(theta_map)+assoc_legendre_deriv(P_lm,li,-mi,ct).*assoc_legendre_deriv(P_lm,lp,mp,ct).*sin(theta_map));
                        term2 = li*(li+1)/k*r.^3.*jh_product(r,k,ks,li,lp,nu,1).*plm_i.*(ellipsoid_function(a,b,c,phi_map,1).*sin(theta_map).*cos(theta_map).*assoc_legendre_deriv(P_lm,lp,mp,ct)+ellipsoid_function(a,b,c,phi_map,2).*sin(phi_map).*cos(phi_map).*plm_p*1i*mp);
                        J(ni,np) = sum(sum(weight_map.*prefactor.*(term1+term2)));
                    end
                end
            end
        end
    case '21'
        for li=1:lmax
            for mi = -li:li
                ni = multi2single_index(1,1,li,mi,lmax);
                for lp=1:lmax
                    for mp=-lp:lp
                        np = multi2single_index(1,1,lp,mp,lmax);
                        prefactor = selection_rules(li,mi,lp,mp,2)*gamma(li,-mi)*gamma(lp,mp)*exp(1i*phi_map*(mp-mi));
                        %make sure we use the right signed associated legendre
                        %polynomial
                        if mp < 0
                            plm_p = P_lm{lp+1,abs(mp)+1}*(-1)^mp*factorial(lp-abs(mp))/factorial(lp+abs(mp));
                        else
                            plm_p = P_lm{lp+1,abs(mp)+1};
                        end
                        if mi > 0
                            plm_i = P_lm{li+1,abs(mi)+1}*(-1)^mi*factorial(li-mi)/factorial(li+mi);
                        else
                            plm_i = P_lm{li+1,abs(mi)+1};
                        end
                        term1 = -1/ks*r.*jh_product(r,k,ks,li,lp,nu,3).*(mi*mp*plm_p.*plm_i./sin(theta_map)+assoc_legendre_deriv(P_lm,li,-mi,ct).*assoc_legendre_deriv(P_lm,lp,mp,ct).*sin(theta_map));
                        term2 = lp*(lp+1)/ks*r.^3.*jh_product(r,k,ks,li,lp,nu,1).*plm_p.*(ellipsoid_function(a,b,c,phi_map,1).*sin(theta_map).^2.*cos(theta_map).*assoc_legendre_deriv(P_lm,li,-mi,ct)-ellipsoid_function(a,b,c,phi_map,2).*sin(phi_map).*cos(phi_map).*sin(theta_map).*plm_i*1i*mp);
                        J(ni,np) = sum(sum(weight_map.*prefactor.*(term1+term2)));
                    end
                end
            end
        end
    case '22'
        for li=1:lmax
            for mi = -li:li
                ni = multi2single_index(1,1,li,mi,lmax);
                for lp=1:lmax
                    for mp=-lp:lp
                        np = multi2single_index(1,1,lp,mp,lmax);
                        prefactor = selection_rules(li,mi,lp,mp,1)*gamma(li,-mi)*gamma(lp,mp)*exp(1i*phi_map*(mp-mi));
                        %make sure we use the right signed associated legendre
                        %polynomial
                        if mp < 0
                            plm_p = P_lm{lp+1,abs(mp)+1}*(-1)^mp*factorial(lp-abs(mp))/factorial(lp+abs(mp));
                        else
                            plm_p = P_lm{lp+1,abs(mp)+1};
                        end
                        if mi > 0
                            plm_i = P_lm{li+1,abs(mi)+1}*(-1)^mi*factorial(li-mi)/factorial(li+mi);
                        else
                            plm_i = P_lm{li+1,abs(mi)+1};
                        end
                        first_term = jh_product(r,k,ks,li,lp,nu,4)/k/ks.*(assoc_legendre_deriv(P_lm,lp,mp,ct)*mi*1i.*plm_i+1i*mp*assoc_legendre_deriv(P_lm,li,-mi,ct).*plm_p);
                        second_term = -ellipsoid_function(a,b,c,phi_map,1)/k/ks.*r.^2.*sin(theta_map).*cos(theta_map).*plm_i.*plm_p.*1i.*(jh_product(r,k,ks,li,lp,nu,3)*mp*li*(li+1)+jh_product(r,k,ks,li,lp,nu,2)*mi*lp*(lp+1));
                        third_term = -ellipsoid_function(a,b,c,phi_map,2)/k/ks.*r.^2.*sin(theta_map).^2.*sin(phi_map).*cos(phi_map).*(jh_product(r,k,ks,li,lp,nu,2).*plm_p*lp*(lp+1).*assoc_legendre_deriv(P_lm,li,-mi,ct)-jh_product(r,k,ks,li,lp,nu,3).*plm_i*li*(li+1).*assoc_legendre_deriv(P_lm,lp,mp,ct));
                        J(ni,np) = sum(sum(weight_map.*prefactor.*(first_term+second_term+third_term)));
                    end
                end
            end
        end
    otherwise
        disp('unsupported J, 11,12,21,22 are valid');
end
disp(toc)
end

function selection_rules = selection_rules(l,m,lp,mp,diag_switch)
if diag_switch == 1
    selection_rules = -(-1)^(m)*(1+(-1)^(mp-m))*(1+(-1)^(lp+l+1));
elseif diag_switch == 2
    selection_rules = -(-1)^(m)*(1+(-1)^(mp-m))*(1+(-1)^(lp+l));
else
    disp('not supported, only valid inputs for diag_switch are 1,2');
end
end

function gamma = gamma(l,m)
gamma = (2*l+1)*factorial(l-m)/4/pi/l/(l+1)/factorial(l+m);
end