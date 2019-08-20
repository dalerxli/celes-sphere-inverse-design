





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
%

function [J11,J12,J21,J22] = compute_J_ellip3(lmax,Ntheta,Nphi,a,b,c,ni,ns,lambda,nu)
tic
%preallocate memory for J
nmax = jmult_max(1,lmax);
J11 = zeros(nmax/2);
J12 = zeros(nmax/2);
J21 = zeros(nmax/2);
J22 = zeros(nmax/2);
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
st = sin(theta_map);
cp = cos(phi_map);
sp = sin(phi_map);
ellip1 = ellipsoid_function(a,b,c,phi_map,1);
ellip2 = ellipsoid_function(a,b,c,phi_map,2);

for li = 1:lmax
    %precompute bessel functions for li,mi and derivative
    b_li = sph_bessel(nu,li,k*r);
    db_li = d1Z_Z_sph_bessel(nu,li,k*r);
    for mi = -li:li
        %compute index ni
        ni = multi2single_index(1,1,li,mi,lmax);
        %get spherical harmonics and derivatives
        if abs(mi) >= 0
            p_limi = P_lm{li+1,abs(mi)+1};
        else
            p_limi = P_lm{li+1,abs(mi)+1}*factorial(li-abs(mi))/factorial(li+abs(mi));            
        end
        dp_limi = assoc_legendre_deriv(P_lm,li,abs(mi),ct);
        for lp = 1:lmax
            %precompute bessel functions for lp,mp, and derivative
            j_lp = sph_bessel(1,lp,ks*r);
            dj_lp = d1Z_Z_sph_bessel(1,lp,ks*r);
            for mp = -lp:lp
                %compute index np
                np = multi2single_index(1,1,lp,mp,lmax);
                %get spherical harmonics and derivatives
                if abs(mp) >= 0
                    p_lpmp = P_lm{lp+1,abs(mp)+1};
                else
                    p_lpmp = P_lm{lp+1,abs(mp)+1}*(-1)^mp*factorial(lp-abs(mp))/factorial(lp+abs(mp));
                end
                dp_lpmp = assoc_legendre_deriv(P_lm,lp,abs(mp),ct);
                
                %setup selection rules for J11,J22
                selection_rules_1122 = selection_rules(li,mi,lp,mp,1);
                %setup selection rules for J12,J21
                selection_rules_1221 = selection_rules(li,mi,lp,mp,2);
                %gamma factor
                gamma_limilpmp = gamma_func(lp,mp)*gamma_func(li,-mi);
                %phi exponential phase factor
                phi_exp = exp(1i*(mp-mi)*phi_map);
                
                %compute J11,J22
                if selection_rules_1122 ~= 0
                    prefactor = selection_rules_1122*gamma_limilpmp*phi_exp;
                    J11(ni,np) = sum(sum(1i*weight_map.*r.^2.*prefactor.*j_lp.*b_li.*(p_lpmp*mp.*dp_limi+p_limi*mi.*dp_lpmp)));
                    J22(ni,np) = sum(sum(weight_map.*prefactor.*(1i/k/ks*dj_lp.*db_li.*(dp_lpmp*mi.*p_limi+dp_limi*mp.*p_lpmp)-1i/ks/k*r.^2.*ellip1.*st.*ct.*(li*(li+1)*mp.*dj_lp.*p_lpmp.*b_li.*p_limi+lp*(lp+1)*mi.*j_lp.*db_li.*p_limi.*p_lpmp)-r.^2.*ellip2.*sp.*cp.*st.^2/ks/k.*(lp*(lp+1).*j_lp.*p_lpmp.*db_li.*dp_limi-li*(li+1)/ks/k.*dj_lp.*dp_lpmp.*b_li.*p_limi))));
                else
                    J11(ni,np) = 0;
                    J22(ni,np) = 0;
                end
                %compute J12,J21
                if selection_rules_1221 ~= 0
                    prefactor = selection_rules_1221*gamma_limilpmp*phi_exp;
                    J12(ni,np) = sum(sum(weight_map.*prefactor.*j_lp.*r.*(db_li/k.*(-mp*mi./st.*p_limi.*p_lpmp-st.*dp_lpmp.*dp_limi)+li*(li+1)*r.^3.*b_li/k.*p_limi.*st.*(ellip1.*dp_lpmp.*ct.*st+ellip2.*sp.*cp*mp.*p_lpmp))));
                    J21(ni,np) = sum(sum(weight_map.*prefactor.*b_li.*r.*(dj_lp/ks.*(-mp*mi./st.*p_limi.*p_lpmp-st.*dp_lpmp.*dp_limi)+lp*(lp+1)*r.^3.*j_lp/ks.*p_lpmp.*st.*(ellip1.*dp_limi.*ct.*st-ellip2.*sp.*cp*mi.*p_limi))));
                else
                    J12(ni,np) = 0;
                    J21(ni,np) = 0;
                end
            end
        end
    end
end
toc
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

function gamma = gamma_func(l,m)
gamma = sqrt((2*l+1)*factorial(l-m)/4/pi/l/(l+1)/factorial(l+m));
end
                    
                
                