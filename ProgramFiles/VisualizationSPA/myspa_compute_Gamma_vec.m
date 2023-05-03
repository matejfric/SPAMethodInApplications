function [Gamma,it_in] = myspa_compute_Gamma_vec(S,Gamma,X)
%COMPUTE_GAMMA Adamar Gamma problem
%
% X     ...D x T
% S     ...D x K
% Gamma ...K x T

[K,T] = size(Gamma);

spgoptions = spgOptions();
spgoptions.maxit = 5e2;
spgoptions.debug = false;
spgoptions.myeps = 1e-8;
spgoptions.alpha_min = 1e-6;
spgoptions.alpha_max = 1e6;

Gamma0 = Gamma;

H1 = (S'*S);
H2 = S'*X;

f = @(Gamma) f_spg(Gamma,X,S,T);
g = @(Gamma) g_spg(Gamma,T,H1,H2);
p = @(Gamma) projection_simplex(Gamma);

f_old = f(Gamma0);
[Gamma,it_in] = spg_vec(@(Gamma) f(Gamma),@(Gamma) g(Gamma),@(Gamma) p(Gamma),Gamma0,spgoptions);
f_new = f(Gamma);

if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
    keyboard
end

end

function L = f_spg(Gamma,X,S,Tcoeff)
%F_SPG Objective function

L = (1/Tcoeff)*sum((X - S*Gamma).^2,1);

end

function G = g_spg(Gamma,Tcoeff,H1,H2)
%G_SPG Gradient function

G = (2/Tcoeff)*(H1*Gamma - H2);

end

